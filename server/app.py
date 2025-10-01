import os
import atexit
import base64
import cv2
import numpy as np
import re
import random
import smtplib
import ssl
import time
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from deepface import DeepFace
from flask import Flask, jsonify, render_template, request, send_from_directory
from gotrue.errors import AuthApiError
from pytz import timezone
from supabase import create_client
from concurrent.futures import ThreadPoolExecutor

from itsdangerous import URLSafeTimedSerializer

# Load environment variables from the .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

print("--- Server starting up ---")

#  Initialize the Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

# Configure Supabase clients for database and auth operations
url: str = os.environ.get("SUPABASE_URL")
anon_key: str = os.environ.get("SUPABASE_KEY")
service_key: str = os.environ.get("SUPABASE_SERVICE_KEY")
# For client-side operations (e.g., login)
supabase_anon = create_client(url, anon_key)
supabase_admin = create_client(url, service_key)  # For server-side admin


campaign_serializer = URLSafeTimedSerializer(
    app.secret_key, salt='campaign-token')

# Load email credentials for the notification system
EMAIL_SENDER = os.environ.get("EMAIL_SENDER_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_SENDER_PASSWORD")

# Pre-load and warm-up the AI model with SSD backend
print("Warming up face recognition model (VGG-Face with SSD)...")
try:
    # Creates a dummy image to pass through the model on startup.
    dummy_image = np.zeros((160, 160, 3), dtype=np.uint8)
    DeepFace.represent(dummy_image, model_name="VGG-Face",
                       detector_backend="ssd", enforce_detection=False)
    print("Face recognition model warmed up successfully.")
except Exception as e:
    print(f"Error during model warmup: {str(e)}")


# TIME MACHINE: Global cache and helper function for time control
system_time_settings = {
    'override_enabled': False,
    'simulated_start_time': None,
    'real_time_at_set': None,
    'last_checked': None
}


def get_system_time():
    sast = timezone('Africa/Johannesburg')
    real_now = datetime.now(sast)

    if not system_time_settings['last_checked'] or (real_now - system_time_settings['last_checked']).seconds > 10:
        try:
            res = supabase_admin.table('system_settings').select(
                '*').eq('id', 1).single().execute()
            if res.data:
                new_override_status = res.data['override_enabled']
                sim_dt_str = res.data.get('simulated_datetime')
                new_sim_time = datetime.fromisoformat(
                    sim_dt_str) if sim_dt_str else None

                if system_time_settings['simulated_start_time'] != new_sim_time or system_time_settings['override_enabled'] != new_override_status:
                    system_time_settings['simulated_start_time'] = new_sim_time
                    system_time_settings['real_time_at_set'] = real_now

                system_time_settings['override_enabled'] = new_override_status
            system_time_settings['last_checked'] = real_now
        except Exception as e:
            print(
                f"TIME_OVERRIDE_ERROR: Could not fetch settings. Defaulting to real time. {e}")
            system_time_settings['override_enabled'] = False

    if system_time_settings['override_enabled'] and system_time_settings['simulated_start_time'] and system_time_settings['real_time_at_set']:
        real_time_elapsed = real_now - system_time_settings['real_time_at_set']
        utc_sim_start = system_time_settings['simulated_start_time']
        sast_sim_start = utc_sim_start.astimezone(sast)
        current_simulated_time = sast_sim_start + real_time_elapsed
        return current_simulated_time

    return real_now
# END TIME MACHINE SECTION

# HELPER FUNCTIONS


def get_face_embedding(image: np.ndarray) -> list:
    try:
        # First try with strict detection
        embedding_objs = DeepFace.represent(
            img_path=image, model_name="VGG-Face", detector_backend="ssd",
            enforce_detection=True, align=True
        )
        return embedding_objs[0]["embedding"] if embedding_objs else None
    except Exception as e:
        print(f"Strict face detection failed: {e}")
        try:
            # Fallback with relaxed detection
            embedding_objs = DeepFace.represent(
                img_path=image, model_name="VGG-Face", detector_backend="ssd",
                enforce_detection=False, align=True
            )
            return embedding_objs[0]["embedding"] if embedding_objs else None
        except Exception as e2:
            print(f"Relaxed face detection also failed: {e2}")
            return None


def cleanup_failed_registration(user_id):
    """Clean up failed registration by deleting user and profile data"""
    try:
        supabase_admin.auth.admin.delete_user(user_id)
        supabase_admin.table('students').delete().eq('id', user_id).execute()
        print(f"Cleaned up failed registration for user {user_id}")
    except Exception as e:
        print(f"Cleanup failed for user {user_id}: {e}")


def send_email(recipient, subject, body):
    try:
        em = EmailMessage()
        em['From'] = EMAIL_SENDER
        em['To'] = recipient
        em['Subject'] = subject
        em.set_content(body)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, recipient, em.as_string())
        print(f"Email sent successfully to {recipient}")
        return True
    except Exception as e:
        print(f"EMAIL FAILED for {recipient}: {str(e)}")
        return False


def check_and_send_at_risk_warning(student_id, module_id):
    """
    Checks if a student has crossed an at-risk threshold (16, 18, or 21 absences)
    and sends them a warning email if they haven't received one for that level yet.
    """
    try:
        # Count total absences for this student in this module
        absences_res = supabase_admin.table('attendance_records').select(
            'id', count='exact'
        ).eq('student_id', student_id).eq('module_id', module_id).eq('status', 'absent').execute()

        total_absences = absences_res.count

        # Define the threshold levels
        thresholds = [16, 18, 21]

        for threshold in thresholds:
            if total_absences == threshold:
                # Check if we've already sent a warning for this level
                existing_warning = supabase_admin.table('at_risk_warnings').select(
                    'id'
                ).eq('student_id', student_id).eq('module_id', module_id).eq(
                    'warning_level', threshold
                ).execute()

                if existing_warning.data:
                    # Already sent this warning, skip
                    continue

                # Fetch student and module details
                student_res = supabase_admin.table('students').select(
                    'full_name, email, module_code'
                ).eq('id', student_id).single().execute()

                if not student_res.data:
                    print(f"Could not find student {student_id}")
                    continue

                student_name = student_res.data['full_name']
                student_email = student_res.data['email']
                module_code = student_res.data['module_code']

                # Fetch lecturer details
                lecturer_res = supabase_admin.table('modules').select(
                    'lecturers(full_name)'
                ).eq('id', module_id).single().execute()

                lecturer_name = "Your Lecturer"
                if lecturer_res.data and lecturer_res.data.get('lecturers'):
                    lecturer_name = lecturer_res.data['lecturers']['full_name']

                # Calculate current mark based on absences
                TOTAL_SEMESTER_CLASSES = 50
                present_count = TOTAL_SEMESTER_CLASSES - total_absences
                current_mark = (present_count / TOTAL_SEMESTER_CLASSES) * 100

                # Send appropriate email based on threshold
                if threshold == 16:
                    subject = f"Check-in: We miss you in {module_code}"
                    body = (
                        f"Hi {student_name},\n\n"
                        f"I wanted to reach out because I've noticed you've missed {total_absences} classes in {module_code} so far this semester.\n\n"
                        f"Your current attendance puts you at {current_mark:.0f}% for the module. You've dropped slightly below our 70% target, but you're still above the 60% pass threshold. However, if you miss 5 more classes, your mark will drop below 60% and you'll be at risk of failing.\n\n"
                        "I believe in your ability to succeed, and I'm here to support you. If there's anything affecting your attendance - whether it's personal challenges, academic struggles, or anything else - please don't hesitate to reach out to me.\n\n"
                        "You can also contact DUT Student Counselling confidentially at: counselling@dut.ac.za\n\n"
                        "Let's work together to get you back on track.\n\n"
                        f"Best regards,\n{lecturer_name}"
                    )
                elif threshold == 18:
                    subject = f"Urgent: Your attendance in {module_code} needs attention"
                    body = (
                        f"Hi {student_name},\n\n"
                        f"This is an important message about your progress in {module_code}.\n\n"
                        f"You have now missed {total_absences} classes, which means your current mark is {current_mark:.0f}%. You are very close to falling below the 60% pass threshold.\n\n"
                        "If you miss just 3 more classes, you will fail the module. I'm reaching out because I want to help you avoid this outcome.\n\n"
                        "Please consider this a serious warning. I strongly encourage you to:\n"
                        "- Attend all remaining classes\n"
                        "- Meet with me to discuss any challenges you're facing\n"
                        "- Reach out to Student Counselling if you need support: counselling@dut.ac.za\n\n"
                        "It's not too late to turn things around, but you need to act now.\n\n"
                        f"Sincerely,\n{lecturer_name}"
                    )
                elif threshold == 21:
                    subject = f"CRITICAL: You are at risk of failing {module_code}"
                    body = (
                        f"Hi {student_name},\n\n"
                        f"This is a critical notice regarding your enrollment in {module_code}.\n\n"
                        f"You have missed {total_absences} classes, which means your current mark is {current_mark:.0f}% - below the 60% pass threshold. You are now at risk of failing this module.\n\n"
                        "This is an urgent situation that requires immediate action. I need you to:\n\n"
                        "1. Contact me immediately to discuss your situation\n"
                        "2. Attend every remaining class without exception\n"
                        "3. Consider whether you may qualify for special consideration (illness, family emergency, etc.)\n\n"
                        "If you're facing serious challenges that have affected your attendance, you may be able to submit a special consideration request through the system. Please reach out to me or Student Counselling (counselling@dut.ac.za) for guidance.\n\n"
                        "I want to help you succeed, but you must take action now.\n\n"
                        f"Urgent regards,\n{lecturer_name}"
                    )

                # Send the email
                email_sent = send_email(student_email, subject, body)

                if email_sent:
                    # Record that we sent this warning
                    supabase_admin.table('at_risk_warnings').insert({
                        'student_id': student_id,
                        'module_id': module_id,
                        'warning_level': threshold,
                        'total_absences_at_warning': total_absences
                    }).execute()

                    print(
                        f"Sent {threshold}-absence warning to {student_name}")

    except Exception as e:
        print(f"--- ERROR in check_and_send_at_risk_warning: {str(e)} ---")


def get_current_class_session():
    try:
        now = get_system_time()
        day_of_week = now.weekday() + 1
        current_time = now.time()
        res = supabase_admin.table('class_timetable').select(
            '*').eq('day_of_week', day_of_week).execute()
        if not res.data:
            return None

        for session in res.data:
            start_time = datetime.strptime(
                session['start_time'], '%H:%M:%S').time()
            attendance_end_time = (datetime.combine(
                get_system_time().date(), start_time) + timedelta(minutes=30)).time()
            if start_time <= current_time <= attendance_end_time:
                return session
        return None
    except Exception as e:
        print(f"Error checking class session: {e}")
        return None


def backfill_missed_attendance():
    """
    Runs on server startup to find past lecture dates with no attendance records
    and marks all enrolled students as absent for those dates.
    """
    print("Starting historical attendance check...")
    try:
        today = get_system_time().date()

        # 1. Get all lectures scheduled before today
        past_lectures_res = supabase_admin.table('lecture_schedules').select(
            'lecture_date, module_id, modules(module_code)'
        ).lt('lecture_date', today.isoformat()).execute()

        if not past_lectures_res.data:
            print("No past lectures found to check.")
            return

        for lecture in past_lectures_res.data:
            lecture_date_str = lecture['lecture_date']
            lecture_date = date.fromisoformat(lecture_date_str)
            module_id = lecture['module_id']
            module_code = lecture['modules']['module_code']

            # 2. For each past lecture, check if ANY attendance was recorded on that day
            records_exist_res = supabase_admin.table('attendance_records').select(
                'id', count='exact'
            ).eq('module_id', module_id).gte(
                'created_at', f"{lecture_date_str}T00:00:00"
            ).lte(
                'created_at', f"{lecture_date_str}T23:59:59"
            ).execute()

            # 3. If NO records exist, it means the server was offline. We need to backfill.
            if records_exist_res.count == 0:
                print(
                    f"-> Found missing attendance for {module_code} on {lecture_date_str}. Backfilling records...")

                # 4. Get all students enrolled in this module
                enrolled_students_res = supabase_admin.table('students').select(
                    'id').eq('module_code', module_code).execute()

                if not enrolled_students_res.data:
                    print(
                        f"   - No students enrolled in {module_code}. Skipping.")
                    continue

                # 5. Create 'absent' records for all enrolled students
                absent_records_to_insert = []
                for student in enrolled_students_res.data:
                    absent_records_to_insert.append({
                        'student_id': student['id'],
                        'module_id': module_id,
                        'status': 'absent',
                        # Set the creation time to the end of that day for historical accuracy
                        'created_at': f"{lecture_date_str}T17:00:00"
                    })

                if absent_records_to_insert:
                    supabase_admin.table('attendance_records').insert(
                        absent_records_to_insert).execute()
                    print(
                        f"   - Successfully created {len(absent_records_to_insert)} absent records.")
            else:
                print(
                    f"Attendance for {module_code} on {lecture_date_str} is already recorded.")

    except Exception as e:
        print(f"--- ERROR during attendance backfill: {str(e)} ---")

# PAGE RENDERING ROUTES


@app.route('/')
def page_home(): return render_template('home.html')


@app.route('/register')
def page_register(): return render_template('register.html')


@app.route('/complete-registration')
def page_complete_registration(): return render_template(
    'complete_registration.html')


@app.route('/login')
def page_login(): return render_template(
    'login.html', supabase_url=url, supabase_key=anon_key)


@app.route('/dashboard')
def dashboard(): return render_template(
    'dashboard.html', supabase_url=url, supabase_key=anon_key)


@app.route('/manage-schedule')
def page_manage_schedule(): return render_template(
    'manage_schedule.html', supabase_url=url, supabase_key=anon_key)


@app.route('/manage-campaigns')
def page_manage_campaigns(): return render_template(
    'manage_campaigns.html', supabase_url=url, supabase_key=anon_key)


@app.route('/manage-submissions')
def page_manage_submissions(): return render_template(
    'manage_submissions.html', supabase_url=url, supabase_key=anon_key)


@app.route('/success')
def page_success(): return render_template('success.html')


@app.route('/attendance')
def page_attendance(): return render_template('attendance.html')


@app.route('/apology')
def page_apology_gateway(): return render_template(
    'apology_gateway.html', supabase_url=url, supabase_key=anon_key)


@app.route('/apology/submit')
def page_apology_form(): return render_template(
    'apology_form.html', supabase_url=url, supabase_key=anon_key)


@app.route('/models/<path:filename>')
def serve_model(filename): return send_from_directory(
    'static/models', filename)


@app.route('/campaign/respond')
def page_campaign_respond():

    return render_template('respond.html')

# AUTOMATED EMAIL & BACKGROUND TASK FUNCTIONS


def send_daily_absence_emails(target_date):
    target_date_str = target_date.isoformat()
    print(f"--- Running absence check for date: {target_date_str} ---")
    try:
        lectures_res = supabase_admin.table('lecture_schedules').select(
            'id, planned_topic, module_id, modules(module_code, lecturers(full_name))'
        ).eq('lecture_date', target_date_str).execute()

        if not lectures_res.data:
            print(f"No lectures were scheduled for {target_date_str}.")
            return

        for lecture in lectures_res.data:
            module_id = lecture['module_id']
            lecturer_name = lecture['modules']['lecturers']['full_name']
            topic = lecture['planned_topic']
            module_code = lecture['modules']['module_code']

            absent_students_res = supabase_admin.table('attendance_records').select(
                '*, students(full_name, email)'
            ).eq('module_id', module_id).eq('status', 'absent').gte(
                'created_at', f"{target_date_str}T00:00:00"
            ).lte(
                'created_at', f"{target_date_str}T23:59:59"
            ).execute()

            if not absent_students_res.data:
                print(
                    f"No students were marked absent for {module_code} on {target_date_str}.")
                continue

            for record in absent_students_res.data:
                student_info = record.get('students')
                if not student_info or not student_info.get('email'):
                    print(
                        f"Skipping student with ID {record.get('student_id')} due to missing info.")
                    continue

                subject = f"Catch-up for {module_code}"
                body = (
                    f"Hi {student_info['full_name']},\n\n"
                    f"I noticed you weren't in our class today. We covered the topic: '{topic}'.\n\n"
                    "Please take some time to review the material. If you have any questions, please don't hesitate to reach out.\n\n"
                    f"Best regards,\n{lecturer_name}"
                )
                send_email(student_info['email'], subject, body)

        print(
            f"--- Absence check for {target_date_str} completed successfully. ---")
    except Exception as e:
        print(
            f"--- ERROR in send_daily_absence_emails for date {target_date_str}: {str(e)} ---")


def send_decision_email(submission_id):
    try:
        res = supabase_admin.table('apology_submissions').select(
            '*, students!inner(full_name, email, module_code)'
        ).eq('id', submission_id).single().execute()

        if not res.data or not res.data.get('students'):
            print(
                f"EMAIL_ERROR: No submission or linked student found for ID {submission_id}")
            return False

        submission = res.data
        student_info = submission['students']
        student_module_code = student_info.get('module_code')

        if not student_module_code:
            print(
                f"EMAIL_ERROR: Student {student_info.get('full_name')} has no module_code assigned.")
            return False

        module_res = supabase_admin.table('modules').select(
            'lecturers!inner(full_name)'
        ).eq('module_code', student_module_code).single().execute()

        lecturer_name = "Your Lecturer"
        if module_res.data and module_res.data.get('lecturers'):
            lecturer_name = module_res.data['lecturers']['full_name']

        student_email = student_info['email']
        student_name = student_info['full_name']
        assessment = submission['assessment_name']
        status = submission['status']
        reason = submission['decision_reason']
        subject = f"Update on your submission for {assessment}"

        if status == 'Approved':
            body = (
                f"Hi {student_name},\n\n"
                f"Good news! Your special consideration request for '{assessment}' has been approved.\n\n"
                f"Reason: {reason}\n\n"
                f"I will be in touch with you shortly regarding the arrangements for your supplementary assessment.\n\n"
                f"Best regards,\n{lecturer_name}"
            )
        else:  # Rejected
            body = (
                f"Hi {student_name},\n\n"
                f"This email is to inform you that your special consideration request for '{assessment}' has been rejected.\n\n"
                f"Reason: {reason}\n\n"
                f"If you would like to discuss this further, please do not hesitate to contact me.\n\n"
                f"Sincerely,\n{lecturer_name}"
            )

        return send_email(student_email, subject, body)
    except Exception as e:
        print(f"--- ERROR in send_decision_email: {str(e)} ---")
        return False


def send_weekly_summary_emails():
    from concurrent.futures import ThreadPoolExecutor
    from collections import defaultdict

    today = get_system_time().date()
    start_of_last_week = today - timedelta(days=today.weekday() + 7)
    end_of_last_week = today - timedelta(days=today.weekday() + 1)
    print(
        f"--- Running weekly summary for {start_of_last_week} to {end_of_last_week} ---")

    try:
        modules_res = supabase_admin.table('modules').select(
            'id, module_code, lecturers(full_name)'
        ).execute()
        if not modules_res.data:
            return

        for module in modules_res.data:
            module_id = module['id']
            module_code = module['module_code']
            lecturer_name = module['lecturers']['full_name']

            # Get total lectures scheduled
            lectures_this_week_res = supabase_admin.table('lecture_schedules').select(
                'id', count='exact'
            ).eq('module_id', module_id).gte(
                'lecture_date', start_of_last_week.isoformat()
            ).lte(
                'lecture_date', end_of_last_week.isoformat()
            ).execute()
            total_lectures_this_week = lectures_this_week_res.count

            if total_lectures_this_week == 0:
                continue

            # BULK QUERY: Get ALL attendance records for the week
            weekly_records_res = supabase_admin.table('attendance_records').select(
                'status, student_id, students(full_name, email)'
            ).eq('module_id', module_id).gte(
                'created_at', start_of_last_week.isoformat()
            ).lte(
                'created_at', f"{end_of_last_week.isoformat()}T23:59:59"
            ).execute()

            if not weekly_records_res.data:
                continue

            # Process in memory: count attendance per student
            student_data = defaultdict(
                lambda: {'present_count': 0, 'details': None})

            for record in weekly_records_res.data:
                student_id = record['student_id']
                student_data[student_id]['details'] = record['students']
                if record['status'] == 'present':
                    student_data[student_id]['present_count'] += 1

            # Separate into two groups
            perfect_attendance = []
            zero_attendance = []

            for student_id, data in student_data.items():
                details = data['details']
                if not details or not details.get('email'):
                    continue

                present_count = data['present_count']

                if present_count == total_lectures_this_week:
                    perfect_attendance.append({
                        'email': details['email'],
                        'subject': f"Great work in {module_code} last week!",
                        'body': (
                            f"Hi {details['full_name']},\n\n"
                            f"Well done! My records show you attended all our {module_code} classes last week. "
                            f"Your commitment is fantastic to see.\n\n"
                            "Keep up the great work!\n\n"
                            f"Best regards,\n{lecturer_name}"
                        )
                    })
                elif present_count == 0:
                    zero_attendance.append({
                        'email': details['email'],
                        'subject': f"Checking in regarding {module_code}",
                        'body': (
                            f"Hi {details['full_name']},\n\n"
                            f"I noticed you were not in any of our {module_code} classes last week "
                            f"and wanted to check in.\n\n"
                            "We miss you in class. If you are facing any challenges, please know you don't have to go through them alone. "
                            "You can reply to this email to talk to me, or confidentially reach out to DUT Student Counselling for support at: counselling@dut.ac.za\n\n"
                            "I am here to help you succeed, so please let me know if there's anything I can do.\n\n"
                            f"Sincerely,\n{lecturer_name}"
                        )
                    })

            # Helper function to send a single email
            def send_single_email(email_data):
                return send_email(email_data['email'], email_data['subject'], email_data['body'])

            # Send emails in parallel with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=5) as executor:
                if perfect_attendance:
                    print(
                        f"Sending {len(perfect_attendance)} perfect attendance emails in parallel...")
                    list(executor.map(send_single_email, perfect_attendance))

                if zero_attendance:
                    print(
                        f"Sending {len(zero_attendance)} zero attendance emails in parallel...")
                    list(executor.map(send_single_email, zero_attendance))

        print("--- Weekly summary check completed successfully. ---")
    except Exception as e:
        print(f"--- ERROR in send_weekly_summary_emails: {str(e)} ---")


def send_campaign_emails(campaign_id):
    """
    Fetches a campaign and its target students, then sends the emails immediately.
    """
    try:
        print(
            f"--- TRIGGERING IMMEDIATE SEND for campaign ID: {campaign_id} ---")

        # 1. Fetch the campaign details
        campaign_res = supabase_admin.table('campaigns').select(
            '*').eq('id', campaign_id).single().execute()
        if not campaign_res.data:
            print(
                f"EMAIL_ERROR: Could not find campaign with ID {campaign_id}.")
            return
        campaign = campaign_res.data

        # 2. Fetch the target students (this can be made more dynamic later)
        students_res = supabase_admin.table('students').select(
            'id, full_name, email').eq('module_code', 'SODM401').execute()
        if not students_res.data:
            print(
                f"EMAIL_ERROR: No students found for campaign '{campaign['title']}'.")
            return

        # 3. Loop through students and send emails
        for student in students_res.data:
            token_data = {
                'campaign_id': campaign['id'], 'student_id': student['id']}
            token = campaign_serializer.dumps(token_data)
            base_url = "http://127.0.0.1:5000"
            response_link = f"{base_url}/campaign/respond?token={token}"

            if campaign['campaign_type'] == 'WEEKLY_QUIZ':
                subject = f"Weekly Progress Quiz is Ready! ({campaign['title']})"
                body = (
                    f"Hi {student['full_name']},\n\n"
                    f"The weekly progress quiz is now available. Please use the unique link below to respond:\n\n"
                    f"Link: {response_link}\n\n"
                    f"Complete the quiz for a chance to win an '{campaign['incentive']}'.\n\n"
                    "Good luck!\nYour Lecturer"
                )
            elif campaign['campaign_type'] == 'FEEDBACK_SURVEY':
                subject = "Your Feedback is Important - Anonymous Survey"
                body = (
                    f"Hi {student['full_name']},\n\n"
                    f"Please take a moment to provide anonymous feedback on the module using the secure link below:\n\n"
                    f"Link: {response_link}\n\n"
                    f"You'll be entered into a draw to win a '{campaign['incentive']}'.\n\n"
                    "Thank you,\nYour Lecturer"
                )
            else:
                continue

            send_email(student['email'], subject, body)

        # 4. Update the campaign status to 'Sent'
        supabase_admin.table('campaigns').update(
            {'status': 'Sent'}).eq('id', campaign['id']).execute()
        print(f"--- Campaign '{campaign['title']}' sent successfully. ---")

    except Exception as e:
        print(f"--- ERROR in send_campaign_emails: {str(e)} ---")


def check_and_process_finished_classes():
    """
    This function runs every few minutes. It uses the Time Machine-aware 
    get_system_time() to check if any classes have recently ended and marks 
    absentees if they haven't been marked already for that session.
    """
    with app.app_context():
        now = get_system_time()
        today_str = now.date().isoformat()
        current_day_of_week = now.weekday() + 1

        print(
            f"SCHEDULER (Interval): Running check at simulated time {now.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            sessions_res = supabase_admin.table('class_timetable').select(
                '*, modules(module_code)'
            ).eq('day_of_week', current_day_of_week).execute()

            if not sessions_res.data:
                return

            for session in sessions_res.data:
                end_time = datetime.strptime(
                    session['end_time'], '%H:%M:%S').time()
                class_end_datetime = now.replace(
                    hour=end_time.hour, minute=end_time.minute, second=0, microsecond=0)

                # Check if class has ended but absentees were not yet processed for today's session
                if now > class_end_datetime:
                    module_id = session['module_id']
                    module_code = session['modules']['module_code']

                    # Check if 'absent' records for this module were already created today after the class ended
                    check_res = supabase_admin.table('attendance_records').select('id', count='exact') \
                        .eq('module_id', module_id) \
                        .eq('status', 'absent') \
                        .gte('created_at', class_end_datetime.isoformat()) \
                        .execute()

                    # If absent records already exist for this session, skip.
                    if check_res.count > 0:
                        continue

                    print(
                        f"SCHEDULER: DETECTED ENDED CLASS! Processing absentees for {module_code}...")

                    enrolled_res = supabase_admin.table('students').select(
                        'id').eq('module_code', module_code).execute()
                    if not enrolled_res.data:
                        continue
                    enrolled_ids = {s['id'] for s in enrolled_res.data}

                    recorded_res = supabase_admin.table('attendance_records').select('student_id') \
                        .eq('module_id', module_id) \
                        .gte('created_at', f"{today_str}T00:00:00") \
                        .lte('created_at', f"{today_str}T23:59:59") \
                        .execute()
                    recorded_ids = {r['student_id'] for r in recorded_res.data}

                    absent_ids = enrolled_ids - recorded_ids

                    if absent_ids:
                        absent_records = []
                        for student_id in absent_ids:
                            absent_records.append({
                                'student_id': student_id,
                                'module_id': module_id,
                                'status': 'absent',
                                'created_at': class_end_datetime.isoformat()
                            })

                        supabase_admin.table('attendance_records').insert(
                            absent_records).execute()
                        print(
                            f"Marked {len(absent_ids)} students as absent for {module_code}.")

                        # NEW: Check each absent student for at-risk warnings
                        for student_id in absent_ids:
                            check_and_send_at_risk_warning(
                                student_id, module_id)

                        # Trigger the email function for the processed date
                        send_daily_absence_emails(target_date=now.date())
                    else:
                        print(f"All students accounted for in {module_code}.")
        except Exception as e:
            print(
                f"--- ERROR in check_and_process_finished_classes: {str(e)} ---")

# API ROUTES


@app.route('/api/register', methods=['POST'])
def api_student_register():
    data = request.get_json()
    email = data.get('email')
    password = "temporary-strong-password-for-verification"
    user_id = None
    try:
        # Get the base URL properly
        base_url = request.url_root.rstrip('/')

        res = supabase_anon.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "email_redirect_to": f"{base_url}/complete-registration"
            }
        })

        if not res.user:
            return jsonify({"error": "Sign up failed."}), 400
        user_id = res.user.id

        # Wait for user creation to be fully committed
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Try to fetch the user to confirm creation
                user_check = supabase_admin.auth.admin.get_user_by_id(user_id)
                if user_check:
                    break
            except:
                retry_count += 1
                time.sleep(1)
                if retry_count == max_retries:
                    raise Exception("User creation confirmation failed")

        profile_data = {
            "full_name": data.get('full_name'),
            "student_number": data.get('student_number'),
            "course": data.get('course'),
            "module_code": data.get('module_code'),
            "year": data.get('year')
        }

        # Retry logic for profile update
        retry_count = 0
        while retry_count < max_retries:
            try:
                update_result = supabase_admin.table('students').update(
                    profile_data).eq('id', user_id).execute()
                if update_result.data:
                    break
            except Exception as update_error:
                retry_count += 1
                if retry_count == max_retries:
                    cleanup_failed_registration(user_id)
                    return jsonify({"error": "Profile update failed. Please try registering again."}), 500
                time.sleep(1)

        return jsonify({"message": "Verification email sent! Please check your inbox and click the link to continue."}), 200

    except AuthApiError as e:
        if 'User already registered' in e.message:
            return jsonify({"error": "This email is already registered."}), 409
        if user_id:
            cleanup_failed_registration(user_id)
        return jsonify({"error": "An authentication error occurred."}), 500
    except Exception as e:
        if user_id:
            cleanup_failed_registration(user_id)
        return jsonify({"error": "A server error occurred."}), 500


@app.route('/api/complete-registration', methods=['POST'])
def api_student_complete_registration():
    auth_header = request.headers.get('Authorization')
    if not auth_header or ' ' not in auth_header:
        return jsonify({"error": "Authorization token is missing."}), 401
    token = auth_header.split(' ')[1]

    try:
        user_res = supabase_anon.auth.get_user(token)
        if not user_res.user:
            return jsonify({"error": "Invalid user token."}), 401

        # Verify user exists in students table first
        existing_user = supabase_admin.table('students').select(
            'id, full_name').eq('id', user_res.user.id).execute()
        if not existing_user.data:
            return jsonify({"error": "User profile not found. Please try registering again."}), 404

        data = request.get_json()
        images_data = data.get('images_data')
        module_code = data.get('module_code')

        if not images_data or not module_code:
            return jsonify({"error": "Missing image data or module code."}), 400

        # Process images with better error reporting
        all_embeddings = []
        failed_images = []

        for i, img_url in enumerate(images_data):
            try:
                img_bytes = base64.b64decode(img_url.split(",")[1])
                img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"Failed to decode image {i + 1}")
                    failed_images.append(i + 1)
                    continue

                emb = get_face_embedding(img)
                if emb:
                    all_embeddings.append(emb)
                else:
                    failed_images.append(i + 1)

            except Exception as e:
                print(f"Failed to process image {i + 1}: {e}")
                failed_images.append(i + 1)

        if len(all_embeddings) == 0:
            return jsonify({"error": f"Could not process any images. Failed images: {failed_images}. Please try again with clearer face images."}), 400
        elif len(all_embeddings) < len(images_data):
            print(
                f"Warning: Only processed {len(all_embeddings)} out of {len(images_data)} images. Failed: {failed_images}")

        # Create master embedding from successful captures
        master_embedding = np.mean(all_embeddings, axis=0).tolist()

        # Update both the embedding and the final module code with retry logic
        update_data = {
            'embedding': master_embedding,
            'module_code': module_code
        }

        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                update_result = supabase_admin.table('students').update(
                    update_data).eq('id', user_res.user.id).execute()
                if update_result.data:
                    break
            except Exception as update_error:
                retry_count += 1
                if retry_count == max_retries:
                    print(
                        f"Final update failed after {max_retries} retries: {update_error}")
                    return jsonify({"error": "Registration completion failed. Please try again."}), 500
                time.sleep(1)

        return jsonify({"message": "Face scan successful! Registration complete."}), 200
    except Exception as e:
        print(f"--- COMPLETE REGISTRATION ERROR: {str(e)} ---")
        return jsonify({"error": "An unexpected server error occurred."}), 500

# Attendance Marking


@app.route('/api/get-current-class', methods=['GET'])
def api_get_current_class():
    session = get_current_class_session()
    current_system_time = get_system_time()
    if session:
        start_time = datetime.strptime(
            session['start_time'], '%H:%M:%S').time()
        attendance_ends_dt = (datetime.combine(
            current_system_time.date(), start_time) + timedelta(minutes=30))
        return jsonify({"isActive": True, "attendance_ends": attendance_ends_dt.isoformat(), "current_time": current_system_time.isoformat()}), 200
    else:
        return jsonify({"isActive": False, "current_time": current_system_time.isoformat()}), 200


@app.route('/api/mark-attendance', methods=['POST'])
def api_mark_attendance():
    active_class = get_current_class_session()
    if not active_class:
        return jsonify({"error": "The attendance window for this class is not open."}), 403

    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({"error": "No image data provided."}), 400
    try:
        header, encoded = data['image_data'].split(",", 1)
        live_image = cv2.imdecode(np.frombuffer(
            base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR)

        embedding = get_face_embedding(live_image)
        if not embedding:
            return jsonify({"error": "No face could be detected in the image."}), 400

        match_res = supabase_admin.rpc('match_student', {'live_embedding': [
                                       float(x) for x in embedding], 'match_threshold': 0.62}).execute()
        if not match_res.data:
            return jsonify({"error": "Face not recognized."}), 404

        best_match = match_res.data[0]
        student_id = best_match['id']
        full_name = best_match['full_name']

        today_start = get_system_time().replace(
            hour=0, minute=0, second=0, microsecond=0)
        existing_record = supabase_admin.table('attendance_records').select('id', count='exact').eq('student_id', student_id).eq(
            'module_id', active_class['module_id']).gte('created_at', today_start.isoformat()).execute()
        if existing_record.count > 0:
            return jsonify({"error": f"{full_name} has already been marked present for this class."}), 409

        attendance_record = {'student_id': student_id,
                             'module_id': active_class['module_id'], 'status': 'present'}
        if system_time_settings['override_enabled']:
            attendance_record['created_at'] = get_system_time().isoformat()

        supabase_admin.table('attendance_records').insert(
            attendance_record).execute()
        return jsonify({"message": f"Attendance marked for {full_name}!", "full_name": full_name}), 200
    except Exception as e:
        print(f"--- ERROR in mark-attendance: {str(e)} ---")
        return jsonify({"error": "An unexpected server error occurred."}), 500

# Apology Submission System


@app.route('/api/send-apology-link', methods=['POST'])
def api_send_apology_link():
    data = request.get_json()
    email = data.get('email')
    if not email:
        return jsonify({"error": "Email is required."}), 400
    student_res = supabase_admin.table('students').select(
        'id').eq('email', email).single().execute()
    if not student_res.data:
        return jsonify({"error": "This email is not registered in the system."}), 404
    try:
        supabase_anon.auth.sign_in_with_otp({"email": email, "options": {
                                            "email_redirect_to": f"{request.host_url}apology/submit"}})
        return jsonify({"message": "A secure link has been sent to your email. Please check your inbox."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to send email: {str(e)}"}), 500


@app.route('/api/submit-apology', methods=['POST'])
def api_handle_apology_submission():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authentication required."}), 401
        token = auth_header.split(' ')[1]
        user_res = supabase_anon.auth.get_user(token)
        if not user_res.user:
            return jsonify({"error": "Invalid or expired token."}), 401

        student_id = user_res.user.id
        assessment_name = request.form.get('assessment_name')
        reason_category = request.form.get('reason_category')
        reason_other_details = request.form.get('reason_other_details')
        if 'proof_file' not in request.files:
            return jsonify({"error": "Proof file is missing."}), 400
        proof_file = request.files['proof_file']
        if proof_file.filename == '':
            return jsonify({"error": "No file selected."}), 400

        file_extension = Path(proof_file.filename).suffix
        file_path_in_bucket = f"{student_id}/{int(time.time())}{file_extension}"
        supabase_admin.storage.from_('proof-uploads').upload(
            file=proof_file.read(),
            path=file_path_in_bucket,
            file_options={"content-type": proof_file.content_type}
        )
        submission_data = {
            'student_id': student_id, 'assessment_name': assessment_name, 'reason_category': reason_category,
            'reason_other_details': reason_other_details, 'proof_file_path': file_path_in_bucket, 'status': 'Pending'
        }
        supabase_admin.table('apology_submissions').insert(
            submission_data).execute()
        return jsonify({"message": "Your submission has been received successfully."}), 201
    except Exception as e:
        print(f"--- ERROR in submit-apology: {str(e)} ---")
        return jsonify({"error": "An unexpected server error occurred."}), 500

# Lecturer Dashboard APIs


@app.route('/api/get-apologies', methods=['GET'])
def api_get_apologies():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authentication required."}), 401
        token = auth_header.split(' ')[1]
        user_res = supabase_anon.auth.get_user(token)
        if not user_res.user:
            return jsonify({"error": "Invalid token."}), 401

        submissions_res = supabase_admin.table('apology_submissions').select(
            '*, students(full_name)').order('created_at', desc=True).execute()
        for sub in submissions_res.data:
            if sub['proof_file_path']:
                signed_url_res = supabase_admin.storage.from_(
                    'proof-uploads').create_signed_url(sub['proof_file_path'], 3600)
                sub['proof_file_url'] = signed_url_res['signedURL']
        return jsonify(submissions_res.data), 200
    except Exception as e:
        print(f"--- ERROR in get-apologies: {str(e)} ---")
        return jsonify({"error": "An unexpected server error occurred."}), 500


@app.route('/api/update-apology-status', methods=['POST'])
def api_update_apology_status():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authentication required."}), 401
        token = auth_header.split(' ')[1]
        user_res = supabase_anon.auth.get_user(token)
        if not user_res.user:
            return jsonify({"error": "Invalid token."}), 401

        data = request.get_json()
        submission_id, new_status, reason = data.get(
            'submission_id'), data.get('status'), data.get('reason')
        if not all([submission_id, new_status, reason]):
            return jsonify({"error": "Missing required fields."}), 400

        supabase_admin.table('apology_submissions').update(
            {'status': new_status, 'decision_reason': reason}).eq('id', submission_id).execute()
        send_decision_email(submission_id)
        return jsonify({"message": f"Submission successfully {new_status.lower()} and student has been notified."}), 200
    except Exception as e:
        print(f"--- ERROR in update-apology-status: {str(e)} ---")
        return jsonify({"error": "An unexpected server error occurred."}), 500


@app.route('/upload-guide', methods=['POST'])
def handle_guide_upload():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing token."}), 401
        token = auth_header.split(' ')[1]
        user_res = supabase_anon.auth.get_user(token)
        if not user_res.user:
            return jsonify({"error": "Invalid token."}), 401

        if 'lecture_guide' not in request.files:
            return jsonify({"error": "No file part."}), 400
        file = request.files['lecture_guide']
        module_id = request.form.get('module_id')

        module_data = supabase_admin.table('modules').select(
            'lecturer_id').eq('id', module_id).single().execute()
        if not module_data.data or module_data.data['lecturer_id'] != str(user_res.user.id):
            return jsonify({"error": "Permission denied."}), 403

        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400
        if file and (file.filename.endswith('.txt') or file.filename.endswith('.md')):
            content = file.stream.read().decode("utf-8")
            schedule_pattern = re.compile(r".*\((\d{4}-\d{2}-\d{2})\):\s*(.*)")
            schedule_entries = [{"module_id": module_id, "lecturer_id": user_res.user.id, "lecture_date": match.group(
                1), "planned_topic": match.group(2).strip()} for line in content.splitlines() if (match := schedule_pattern.match(line))]
            if not schedule_entries:
                return jsonify({"error": "Could not find any valid schedule entries in the file."}), 400

            supabase_admin.table('lecture_schedules').upsert(
                schedule_entries, on_conflict='module_id, lecture_date').execute()
            return jsonify({"message": f"Successfully processed {len(schedule_entries)} lecture topics!"}), 200

        return jsonify({"error": "Invalid file type."}), 400
    except Exception as e:
        print(f"--- ERROR in guide-upload: {str(e)} ---")
        return jsonify({"error": "An unexpected server error occurred."}), 500


@app.route('/api/dashboard-data', methods=['GET'])
def api_get_dashboard_data():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authentication required."}), 401
        token = auth_header.split(' ')[1]
        user_res = supabase_anon.auth.get_user(token)
        if not user_res.user:
            return jsonify({"error": "Invalid token."}), 401

        lecturer_id = user_res.user.id
        module_res = supabase_admin.table('modules').select(
            'id, module_code').eq('lecturer_id', lecturer_id).single().execute()
        if not module_res.data:
            return jsonify({"error": "No module assigned to this lecturer."}), 404
        module_id, module_code = module_res.data['id'], module_res.data['module_code']

        # --- DYNAMIC CALCULATION (NO MORE HARDCODING) ---

        # 1. Get the TOTAL number of scheduled lectures for the entire semester for this module
        total_schedule_res = supabase_admin.table('lecture_schedules').select(
            'id', count='exact'
        ).eq('module_id', module_id).execute()
        total_semester_classes = total_schedule_res.count

        if total_semester_classes == 0:
            return jsonify({"error": "No lectures have been scheduled for this module yet."}), 404

        # 2. Get the number of lectures held up to and including today
        today = get_system_time().date()
        held_schedule_res = supabase_admin.table('lecture_schedules').select(
            'id', count='exact'
        ).eq('module_id', module_id).lte('lecture_date', today.isoformat()).execute()
        classes_held_so_far = held_schedule_res.count

        mark_per_class = 100 / total_semester_classes
        pass_threshold = 60
        # --- END DYNAMIC CALCULATION ---

        student_count_res = supabase_admin.table('students').select(
            'id', count='exact').eq('module_code', module_code).execute()

        timetable_res = supabase_admin.table('class_timetable').select(
            'day_of_week').eq('module_id', module_id).execute()

        students_res = supabase_admin.table('students').select(
            'id, full_name, student_number, attendance_records(count)'
        ).eq('module_code', module_code).eq('attendance_records.status', 'present').execute()

        processed_students = []
        for student in students_res.data:
            present_count = student['attendance_records'][0]['count'] if student['attendance_records'] else 0
            absences_count = classes_held_so_far - present_count  # Calculate absences
            mark_so_far = present_count * mark_per_class
            remaining_classes = total_semester_classes - classes_held_so_far
            max_potential_mark = mark_so_far + \
                (remaining_classes * mark_per_class)

            # CORRECTED: At risk means 16 or more absences (not based on potential mark)
            is_at_risk = absences_count >= 16

            processed_students.append({
                "full_name": student['full_name'],
                "student_number": student['student_number'],
                "is_at_risk": is_at_risk
            })

        # FIXED QUERY: Use an inner join to prevent "Unknown Student" where possible
        # CRITICAL FIX: Added 'full_name' to the students selection
        attendance_res = supabase_admin.table('attendance_records').select(
            'student_id, status, created_at, students!inner(full_name, student_number)'
        ).eq('module_id', module_id).execute()

        pending_subs_res = supabase_admin.table('apology_submissions').select(
            'id', count='exact').eq('status', 'Pending').execute()

        return jsonify({
            "module_id": module_id,
            "module_code": module_code,
            "total_students": student_count_res.count,
            "total_semester_classes": total_semester_classes,
            "classes_held_so_far": classes_held_so_far,
            "class_days": [item['day_of_week'] for item in timetable_res.data],
            "students": processed_students,
            "records": attendance_res.data or [],
            "pending_submissions_count": pending_subs_res.count
        }), 200
    except Exception as e:
        print(f"--- ERROR in get-dashboard-data: {str(e)} ---")
        return jsonify({"error": "An unexpected server error occurred."}), 500

# Campaign Management APIs


@app.route('/api/create-campaign', methods=['POST'])
def api_create_campaign():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing token."}), 401
        token = auth_header.split(' ')[1]
        user_res = supabase_anon.auth.get_user(token)
        if not user_res.user:
            return jsonify({"error": "Invalid token."}), 401

        data = request.get_json()
        campaign_type = data.get('campaign_type')
        questions = data.get('questions', [])
        if not all([campaign_type, questions]):
            return jsonify({"error": "Campaign type and at least one question are required."}), 400

        is_anonymous = (campaign_type == 'FEEDBACK_SURVEY')
        title = "Lecturer Feedback Survey" if is_anonymous else f"Weekly Quiz (Week of {get_system_time().date().strftime('%d %b')})"
        incentive = "1-Month Staff WiFi Pass (x1 Winner)" if is_anonymous else "Assignment Extension Voucher (x5 Winners)"

        campaign_insert_res = supabase_admin.table('campaigns').insert({
            "lecturer_id": user_res.user.id,
            "campaign_type": campaign_type,
            "title": title,
            "incentive": incentive,
            "is_anonymous": is_anonymous,
            "status": 'Sending',
        }).execute()

        new_campaign_id = campaign_insert_res.data[0]['id']

        for q in questions:
            question_data = {
                'campaign_id': new_campaign_id,
                'question_text': q['text'],
                'question_type': q['type']
            }
            if q['type'] == 'multiple_choice' and q.get('correct_answer'):
                question_data['correct_answer'] = q['correct_answer']

            question_insert_res = supabase_admin.table(
                'campaign_questions').insert(question_data).execute()
            new_question_id = question_insert_res.data[0]['id']

            if q['type'] == 'multiple_choice' and q['options']:
                options_to_insert = [
                    {'question_id': new_question_id, 'option_text': opt} for opt in q['options']]
                supabase_admin.table('question_options').insert(
                    options_to_insert).execute()

        send_campaign_emails(new_campaign_id)

        return jsonify({"message": f"{title} created and emails sent successfully!"}), 201
    except Exception as e:
        if 'new_campaign_id' in locals():
            supabase_admin.table('campaigns').update(
                {'status': 'Failed'}).eq('id', new_campaign_id).execute()
        print(f"--- ERROR in create-campaign: {str(e)} ---")
        return jsonify({"error": "An unexpected server error occurred."}), 500


@app.route('/api/get-campaign-for-student', methods=['POST'])
def api_get_campaign_for_student():
    try:
        data = request.get_json()
        token = data.get('token')
        if not token:
            return jsonify({"error": "Invalid link. Token is missing."}), 400

        token_data = campaign_serializer.loads(token, max_age=259200)
        campaign_id = token_data['campaign_id']
        student_id = token_data['student_id']

        existing_response = supabase_admin.table('campaign_participants').select(
            'id').eq('campaign_id', campaign_id).eq('student_id', student_id).execute()
        if existing_response.data:
            return jsonify({"error": "You have already responded to this campaign. Thank you!"}), 409

        campaign_res = supabase_admin.table('campaigns').select(
            '*').eq('id', campaign_id).single().execute()
        if not campaign_res.data:
            return jsonify({"error": "Campaign not found."}), 404

        campaign = campaign_res.data

        questions_res = supabase_admin.table('campaign_questions').select(
            '*').eq('campaign_id', campaign_id).execute()
        campaign['campaign_questions'] = questions_res.data

        for q in campaign['campaign_questions']:
            if q['question_type'] == 'multiple_choice':
                options_res = supabase_admin.table('question_options').select(
                    'option_text').eq('question_id', q['id']).execute()
                q['options'] = [opt['option_text'] for opt in options_res.data]

        return jsonify({"campaign": campaign}), 200
    except Exception as e:
        return jsonify({"error": f"This link may be invalid or expired. {e}"}), 400


@app.route('/api/submit-campaign-response', methods=['POST'])
def api_submit_campaign_response():
    try:
        data = request.get_json()
        token = data.get('token')
        responses = data.get('responses')
        if not token or not responses:
            return jsonify({"error": "Invalid submission."}), 400

        token_data = campaign_serializer.loads(token, max_age=259200)
        campaign_id = token_data['campaign_id']
        student_id = token_data['student_id']

        existing_response = supabase_admin.table('campaign_participants').select(
            'id').eq('campaign_id', campaign_id).eq('student_id', student_id).execute()
        if existing_response.data:
            return jsonify({"error": "You have already submitted a response."}), 409

        score = None
        campaign_type_res = supabase_admin.table('campaigns').select(
            'campaign_type').eq('id', campaign_id).single().execute()

        if campaign_type_res.data and campaign_type_res.data['campaign_type'] == 'WEEKLY_QUIZ':
            score = 0
            questions_res = supabase_admin.table('campaign_questions').select(
                'id, correct_answer').eq('campaign_id', campaign_id).execute()
            correct_answers_map = {q['id']: q['correct_answer']
                                   for q in questions_res.data}

            for question_id, student_answer in responses.items():
                correct_answer = correct_answers_map.get(int(question_id))
                if correct_answer and student_answer == correct_answer:
                    score += 1

        supabase_admin.table('campaign_participants').insert({
            'campaign_id': campaign_id,
            'student_id': student_id
        }).execute()

        supabase_admin.table('campaign_responses').insert({
            'campaign_id': campaign_id,
            'student_id': student_id,
            'response_data': responses,
            'score': score
        }).execute()

        return jsonify({"message": "Your response has been recorded successfully. You are now eligible for the incentive!"}), 201
    except Exception as e:
        return jsonify({"error": f"Could not submit response. The link may have expired. {e}"}), 500


@app.route('/api/get-campaigns', methods=['GET'])
def api_get_campaigns():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing token."}), 401
        token = auth_header.split(' ')[1]
        user_res = supabase_anon.auth.get_user(token)
        if not user_res.user:
            return jsonify({"error": "Invalid token."}), 401

        campaigns_res = supabase_admin.table('campaigns').select(
            "*, campaign_questions(*), campaign_participants(count), campaign_responses(score, response_data, students(full_name, student_number)), vouchers(is_redeemed, students(full_name, student_number))"
        ).eq('lecturer_id', user_res.user.id).order('created_at', desc=True).execute()

        return jsonify(campaigns_res.data), 200
    except Exception as e:
        print(f"--- ERROR in get-campaigns: {str(e)} ---")
        return jsonify({"error": "Server error fetching campaigns."}), 500


@app.route('/api/pick-winners', methods=['POST'])
def api_pick_winners():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing token."}), 401
        token = auth_header.split(' ')[1]
        user_res = supabase_anon.auth.get_user(token)
        if not user_res.user:
            return jsonify({"error": "Invalid token."}), 401

        data = request.get_json()
        campaign_id = data.get('campaign_id')

        campaign_res = supabase_admin.table('campaigns').select(
            '*').eq('id', campaign_id).eq('lecturer_id', user_res.user.id).single().execute()
        if not campaign_res.data:
            return jsonify({"error": "Campaign not found."}), 404

        campaign = campaign_res.data
        participants_res = supabase_admin.table('campaign_participants').select(
            'student_id, students(full_name, email)').eq('campaign_id', campaign_id).execute()
        if not participants_res.data:
            return jsonify({"error": "No students have participated yet."}), 400

        num_winners = int(re.search(r'x(\d+)', campaign['incentive']).group(
            1)) if re.search(r'x(\d+)', campaign['incentive']) else 1
        if len(participants_res.data) < num_winners:
            return jsonify({"error": f"Not enough participants to pick {num_winners} winners."}), 400

        winners = random.sample(participants_res.data, num_winners)

        for winner in winners:
            student_id, student_name, student_email = winner['student_id'], winner[
                'students']['full_name'], winner['students']['email']

            if "Assignment Extension" in campaign['incentive']:
                subject, body, voucher_type = "🎉 Congratulations! You've won an Assignment Extension Voucher!", f"Hi {student_name},\n\nGreat news! For completing the '{campaign['title']}', you've been randomly selected to receive an Assignment Extension Voucher.\n\nThis voucher grants you a 24-hour extension on a single assignment. To redeem it, please forward this email to me when you submit.\n\nWell done and thank you for your participation!\nYour Lecturer", "Assignment Extension"
            elif "WiFi Pass" in campaign['incentive']:
                subject, body, voucher_type = "🎉 Congratulations! You've won a Staff WiFi Pass!", f"Hi {student_name},\n\nGreat news! For completing the '{campaign['title']}', you've been randomly selected to receive a one-month DUT Staff WiFi Pass.\n\nPlease see me after our next class to collect your pass.\n\nThank you for your valuable feedback!\nYour Lecturer", "Staff WiFi Pass"
            else:
                subject, body, voucher_type = "Congratulations! You're a campaign winner!", f"Hi {student_name}, You have won the prize for the '{campaign['title']}' campaign.", "Generic Prize"

            send_email(student_email, subject, body)
            supabase_admin.table('vouchers').insert({"student_id": student_id, "campaign_id": campaign_id, "voucher_type": voucher_type, "expires_on": (
                get_system_time().date() + timedelta(days=30)).isoformat()}).execute()

        supabase_admin.table('campaigns').update(
            {'status': 'COMPLETED'}).eq('id', campaign_id).execute()
        return jsonify({"message": "Winners selected and notified successfully!"}), 200
    except Exception as e:
        print(f"--- ERROR in pick-winners: {str(e)} ---")
        return jsonify({"error": "An unexpected server error occurred."}), 500


@app.route('/api/admin/backfill-all-students', methods=['POST'])
def api_backfill_all_students():
    """
    ADMIN ONLY: Backfills missing attendance records for all students
    who were registered after classes began - ONLY for dates that have already passed
    """
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authentication required."}), 401
        token = auth_header.split(' ')[1]
        user_res = supabase_anon.auth.get_user(token)
        if not user_res.user:
            return jsonify({"error": "Invalid token."}), 401

        lecturer_id = user_res.user.id
        module_res = supabase_admin.table('modules').select(
            'id, module_code'
        ).eq('lecturer_id', lecturer_id).single().execute()

        if not module_res.data:
            return jsonify({"error": "No module found."}), 404

        module_id = module_res.data['id']
        module_code = module_res.data['module_code']

        # Get all students in this module
        students_res = supabase_admin.table('students').select(
            'id'
        ).eq('module_code', module_code).execute()

        if not students_res.data:
            return jsonify({"error": "No students found."}), 404

        # CRITICAL FIX: Only get lectures that have ALREADY HAPPENED
        today = get_system_time().date()
        lectures_res = supabase_admin.table('lecture_schedules').select(
            'lecture_date'
        ).eq('module_id', module_id).lt('lecture_date', today.isoformat()).execute()

        if not lectures_res.data:
            return jsonify({"message": "No past lectures to backfill."}), 200

        backfill_count = 0

        for student in students_res.data:
            student_id = student['id']

            # For each PAST lecture, check if this student has ANY record
            for lecture in lectures_res.data:
                lecture_date = lecture['lecture_date']

                # Check if a record already exists for this student on this date
                existing_res = supabase_admin.table('attendance_records').select('id').eq(
                    'student_id', student_id
                ).eq('module_id', module_id).gte(
                    'created_at', f"{lecture_date}T00:00:00"
                ).lte(
                    'created_at', f"{lecture_date}T23:59:59"
                ).execute()

                # If no record exists, create an absent record
                if not existing_res.data:
                    supabase_admin.table('attendance_records').insert({
                        'student_id': student_id,
                        'module_id': module_id,
                        'status': 'absent',
                        'created_at': f"{lecture_date}T17:00:00"
                    }).execute()
                    backfill_count += 1

        return jsonify({
            "message": f"Backfill complete. Added {backfill_count} historical absence records for past classes only."
        }), 200

    except Exception as e:
        print(f"--- ERROR in backfill: {str(e)} ---")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Manual Trigger & Utility Routes


@app.route('/api/logout', methods=['POST'])
def api_handle_logout():
    try:
        supabase_anon.auth.sign_out()
        return jsonify({"message": "Successfully logged out"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/run-daily-check', methods=['POST'])
def api_run_daily_check():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authentication required."}), 401
        token = auth_header.split(' ')[1]
        user_res = supabase_anon.auth.get_user(token)
        if not user_res.user:
            return jsonify({"error": "Invalid token."}), 401

        send_daily_absence_emails(
            target_date=get_system_time().date() - timedelta(days=1))
        return jsonify({"message": "Daily check for yesterday completed."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-at-risk-check', methods=['POST'])
def api_run_at_risk_check():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authentication required."}), 401
        token = auth_header.split(' ')[1]
        user_res = supabase_anon.auth.get_user(token)
        if not user_res.user:
            return jsonify({"error": "Invalid token."}), 401

        lecturer_id = user_res.user.id
        module_res = supabase_admin.table('modules').select(
            'id, module_code'
        ).eq('lecturer_id', lecturer_id).single().execute()

        if not module_res.data:
            return jsonify({"error": "No module found."}), 404

        module_id = module_res.data['id']
        module_code = module_res.data['module_code']

        students_res = supabase_admin.table('students').select(
            'id, full_name, email'
        ).eq('module_code', module_code).execute()

        if not students_res.data:
            return jsonify({"error": "No students found."}), 404

        # Get schedule info
        total_schedule_res = supabase_admin.table('lecture_schedules').select(
            'id', count='exact'
        ).eq('module_id', module_id).execute()
        TOTAL_SEMESTER_CLASSES = total_schedule_res.count

        today = get_system_time().date()
        held_schedule_res = supabase_admin.table('lecture_schedules').select(
            'id', count='exact'
        ).eq('module_id', module_id).lte('lecture_date', today.isoformat()).execute()
        classes_held = held_schedule_res.count

        MARK_PER_CLASS = 100 / TOTAL_SEMESTER_CLASSES

        # BULK QUERY: Get ALL absences in one go
        all_absences_res = supabase_admin.table('attendance_records').select(
            'student_id'
        ).eq('module_id', module_id).eq('status', 'absent').execute()

        # Count absences per student in Python
        from collections import Counter
        absence_counts = Counter(record['student_id']
                                 for record in all_absences_res.data)

        # Get lecturer name once
        lecturer_res = supabase_admin.table('modules').select(
            'lecturers(full_name)'
        ).eq('id', module_id).single().execute()

        lecturer_name = "Your Lecturer"
        if lecturer_res.data and lecturer_res.data.get('lecturers'):
            lecturer_name = lecturer_res.data['lecturers']['full_name']

        emails_sent = 0
        at_risk_students = []
        email_errors = []

        for student in students_res.data:
            student_id = student['id']
            student_name = student['full_name']
            student_email = student['email']

            # Get absence count from our bulk query results
            total_absences = absence_counts.get(student_id, 0)

            print(f"DEBUG: {student_name} has {total_absences} absences")

            # Check thresholds
            thresholds = [21, 18, 16]
            email_sent = False

            for threshold in thresholds:
                if total_absences >= threshold and not email_sent:
                    print(f"DEBUG: {student_name} meets threshold {threshold}")

                    # Calculate current mark
                    present_count = classes_held - total_absences
                    current_mark = (
                        present_count / TOTAL_SEMESTER_CLASSES) * 100

                    # Determine email content based on threshold
                    if threshold == 21:
                        subject = f"CRITICAL: You are at risk of failing {module_code}"
                        body = (
                            f"Hi {student_name},\n\n"
                            f"This is a critical notice regarding your enrollment in {module_code}.\n\n"
                            f"You have missed {total_absences} classes, which means your current mark is {current_mark:.0f}% - below the 60% pass threshold. You are now at risk of failing this module.\n\n"
                            "This is an urgent situation that requires immediate action. I need you to:\n\n"
                            "1. Contact me immediately to discuss your situation\n"
                            "2. Attend every remaining class without exception\n"
                            "3. Consider whether you may qualify for special consideration (illness, family emergency, etc.)\n\n"
                            "If you're facing serious challenges that have affected your attendance, you may be able to submit a special consideration request through the system. Please reach out to me or Student Counselling (counselling@dut.ac.za) for guidance.\n\n"
                            "I want to help you succeed, but you must take action now.\n\n"
                            f"Urgent regards,\n{lecturer_name}"
                        )
                    elif threshold == 18:
                        subject = f"Urgent: Your attendance in {module_code} needs attention"
                        body = (
                            f"Hi {student_name},\n\n"
                            f"This is an important message about your progress in {module_code}.\n\n"
                            f"You have now missed {total_absences} classes, which means your current mark is {current_mark:.0f}%. You are very close to falling below the 60% pass threshold.\n\n"
                            "If you miss just 3 more classes, you will fail the module. I'm reaching out because I want to help you avoid this outcome.\n\n"
                            "Please consider this a serious warning. I strongly encourage you to:\n"
                            "- Attend all remaining classes\n"
                            "- Meet with me to discuss any challenges you're facing\n"
                            "- Reach out to Student Counselling if you need support: counselling@dut.ac.za\n\n"
                            "It's not too late to turn things around, but you need to act now.\n\n"
                            f"Sincerely,\n{lecturer_name}"
                        )
                    else:  # 16 absences
                        subject = f"Check-in: We miss you in {module_code}"
                        body = (
                            f"Hi {student_name},\n\n"
                            f"I wanted to reach out because I've noticed you've missed {total_absences} classes in {module_code} so far this semester.\n\n"
                            f"Your current attendance puts you at {current_mark:.0f}% for the module. You've dropped slightly below our 70% target, but you're still above the 60% pass threshold. However, if you miss 5 more classes, your mark will drop below 60% and you'll be at risk of failing.\n\n"
                            "I believe in your ability to succeed, and I'm here to support you. If there's anything affecting your attendance - whether it's personal challenges, academic struggles, or anything else - please don't hesitate to reach out to me.\n\n"
                            "You can also contact DUT Student Counselling confidentially at: counselling@dut.ac.za\n\n"
                            "Let's work together to get you back on track.\n\n"
                            f"Best regards,\n{lecturer_name}"
                        )

                    # Attempt to send email
                    try:
                        email_result = send_email(student_email, subject, body)
                        if email_result:
                            emails_sent += 1
                            at_risk_students.append(
                                f"{student_name} ({total_absences} absences - {threshold} threshold)")
                            email_sent = True
                            print(f"SUCCESS: Email sent to {student_name}")
                        else:
                            email_errors.append(
                                f"{student_name}: Email send failed")
                            print(f"FAILED: Email to {student_name} failed")
                    except Exception as e:
                        email_errors.append(f"{student_name}: {str(e)}")
                        print(
                            f"ERROR: Email to {student_name} threw exception: {e}")

                    break  # Only send one email per student

        result_message = f"At-risk check completed. Sent {emails_sent} warning email(s)."
        if email_errors:
            result_message += f" {len(email_errors)} email(s) failed."

        return jsonify({
            "message": result_message,
            "students_contacted": at_risk_students,
            "errors": email_errors if email_errors else None
        }), 200

    except Exception as e:
        print(f"--- ERROR in run-at-risk-check: {str(e)} ---")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# TIME MACHINE API


@app.route('/api/system-time', methods=['GET', 'POST'])
def api_system_time_control():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "Authentication required."}), 401
    token = auth_header.split(' ')[1]
    try:
        user_res = supabase_anon.auth.get_user(token)
        if not user_res.user:
            return jsonify({"error": "Invalid token."}), 401
    except Exception:
        return jsonify({"error": "Invalid token."}), 401

    if request.method == 'GET':
        current_time = get_system_time()
        res = supabase_admin.table('system_settings').select(
            '*').eq('id', 1).single().execute()
        res.data['current_system_time'] = current_time.isoformat()
        return jsonify(res.data), 200

    if request.method == 'POST':
        data = request.get_json()
        supabase_admin.table('system_settings').update({
            'override_enabled': data.get('override_enabled'),
            'simulated_datetime': data.get('simulated_datetime')
        }).eq('id', 1).execute()

        system_time_settings['override_enabled'] = data.get('override_enabled')
        sim_dt_str = data.get('simulated_datetime')
        if sim_dt_str:
            system_time_settings['simulated_start_time'] = datetime.fromisoformat(
                sim_dt_str)
            system_time_settings['real_time_at_set'] = datetime.now(
                timezone('Africa/Johannesburg'))

        system_time_settings['last_checked'] = datetime.now(
            timezone('Africa/Johannesburg'))
        return jsonify({"message": "System time settings updated."}), 200

# AUTOMATED SCHEDULER LOGIC (TIME MACHINE AWARE)


# Global state to track when jobs were last run to prevent duplicates
last_run_times = {"absentee_check": {},
                  "weekly_summary": None, "campaigns": None}


def unified_scheduler_job():
    """
    A single, smart scheduler job that runs every minute and checks if any
    of the system's automated tasks should be triggered.
    """
    with app.app_context():
        now = get_system_time()
        today_str = now.date().isoformat()

        print(
            f"SCHEDULER (Unified): Tick at simulated time {now.strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Check for Finished Classes to Mark Absentees
        try:
            sessions_res = supabase_admin.table('class_timetable').select(
                '*, modules(module_code)').eq('day_of_week', now.weekday() + 1).execute()
            if sessions_res.data:
                for session in sessions_res.data:
                    end_time = datetime.strptime(
                        session['end_time'], '%H:%M:%S').time()
                    class_end_datetime = now.replace(
                        hour=end_time.hour, minute=end_time.minute, second=0, microsecond=0)

                    if now > class_end_datetime and (now - class_end_datetime).seconds <= 300:
                        session_id_for_day = f"{today_str}_{session['id']}"
                        if last_run_times["absentee_check"].get(session_id_for_day) != True:
                            print(
                                f"--> TRIGGER: Marking absentees for {session['modules']['module_code']}")

                            module_id = session['module_id']
                            module_code = session['modules']['module_code']

                            enrolled_res = supabase_admin.table('students').select(
                                'id').eq('module_code', module_code).execute()
                            if not enrolled_res.data:
                                continue
                            enrolled_ids = {s['id'] for s in enrolled_res.data}

                            recorded_res = supabase_admin.table('attendance_records').select('student_id') \
                                .eq('module_id', module_id) \
                                .gte('created_at', f"{today_str}T00:00:00") \
                                .lte('created_at', f"{today_str}T23:59:59") \
                                .execute()
                            recorded_ids = {r['student_id']
                                            for r in recorded_res.data}

                            absent_ids = enrolled_ids - recorded_ids

                            if absent_ids:
                                absent_records = []
                                for student_id in absent_ids:
                                    absent_records.append({
                                        'student_id': student_id,
                                        'module_id': module_id,
                                        'status': 'absent',
                                        'created_at': class_end_datetime.isoformat()
                                    })

                                supabase_admin.table('attendance_records').insert(
                                    absent_records).execute()
                                print(
                                    f"Marked {len(absent_ids)} students as absent for {module_code}.")

                                # Check each absent student for at-risk warnings
                                for student_id in absent_ids:
                                    check_and_send_at_risk_warning(
                                        student_id, module_id)

                                send_daily_absence_emails(
                                    target_date=now.date())

                            last_run_times["absentee_check"][session_id_for_day] = True
        except Exception as e:
            print(
                f"--- ERROR in unified scheduler (absentee check): {str(e)} ---")

        # 2. Check for Weekly Summaries (Sundays after 6 PM)
        try:
            if now.weekday() == 6 and now.hour >= 18:
                if last_run_times["weekly_summary"] != today_str:
                    print("--> TRIGGER: Sending weekly summary emails...")
                    send_weekly_summary_emails()
                    last_run_times["weekly_summary"] = today_str
        except Exception as e:
            print(
                f"--- ERROR in unified scheduler (weekly summary): {str(e)} ---")


# Initialize and Start Scheduler
scheduler = BackgroundScheduler(timezone='Africa/Johannesburg')
scheduler.add_job(unified_scheduler_job, 'interval',
                  minutes=1, id='unified_scheduler_job')
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    print("--- Running one-time attendance backfill check ---")
    backfill_missed_attendance()
    print("--- Backfill check complete ---")

    print("Starting Flask development server...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
