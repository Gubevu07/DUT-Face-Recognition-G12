// This function will run when the page is loaded
document.addEventListener('DOMContentLoaded', () => {

    // --- 1. SHARED DOM ELEMENTS ---
    // References to elements that appear on every page
    const dom = {
        logoutBtn: document.getElementById('logout-btn'),
        themeToggle: document.getElementById('theme-toggle'),
        mobileMenuBtn: document.getElementById('mobile-menu-btn'),
        sidebarNav: document.getElementById('sidebar-nav'),
        submissionsCounter: document.getElementById('submissions-counter')
    };

    // --- 2. SUPABASE INITIALIZATION ---
    // This should be the only place you define your Supabase client
    const SUPABASE_URL = '{{ supabase_url }}';
    const SUPABASE_ANON_KEY = '{{ supabase_key }}';
    let supabaseClient;

    // A check to make sure the keys are being passed correctly
    {% raw %}
    if (!SUPABASE_URL || !SUPABASE_ANON_KEY || SUPABASE_URL.includes('{{')) {
        console.error("Supabase keys not passed to template properly.");
        // Optionally, hide elements that require Supabase
        return;
    }
    {% endraw %}
    supabaseClient = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);


    // --- 3. SUBMISSIONS COUNTER LOGIC ---
    // A dedicated function to fetch and update the counter
    async function fetchSubmissionCount() {
        // Ensure the counter element exists on the page before running a query
        if (!dom.submissionsCounter) {
            return;
        }

        try {
            const { count, error } = await supabaseClient
                .from('apology_submissions')
                .select('*', { count: 'exact', head: true })
                .eq('status', 'Pending');

            if (error) {
                // Don't show an error, just log it for debugging
                console.error("Error fetching submission count:", error.message);
                return;
            }

            if (count > 0) {
                dom.submissionsCounter.innerText = count;
                dom.submissionsCounter.style.display = 'inline-block';
            } else {
                dom.submissionsCounter.style.display = 'none';
            }

        } catch (err) {
            console.error("A network or other error occurred:", err);
        }
    }


    // --- 4. SHARED EVENT LISTENERS ---

    // Theme Toggle Logic
    if (dom.themeToggle) {
        const currentTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', currentTheme);
        dom.themeToggle.innerHTML = currentTheme === 'dark' ? '<i class="bx bx-sun"></i>' : '<i class="bx bx-moon"></i>';

        dom.themeToggle.addEventListener('click', () => {
            const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            if (isDark) {
                document.documentElement.removeAttribute('data-theme');
                localStorage.setItem('theme', 'light');
                dom.themeToggle.innerHTML = '<i class="bx bx-moon"></i>';
            } else {
                document.documentElement.setAttribute('data-theme', 'dark');
                localStorage.setItem('theme', 'dark');
                dom.themeToggle.innerHTML = '<i class="bx bx-sun"></i>';
            }
        });
    }

    // Mobile Menu Logic
    if (dom.mobileMenuBtn && dom.sidebarNav) {
        dom.mobileMenuBtn.addEventListener('click', () => {
            dom.sidebarNav.classList.toggle('active');
        });
    }

    // Logout Button Logic
    if (dom.logoutBtn) {
        dom.logoutBtn.addEventListener('click', async () => {
            await supabaseClient.auth.signOut();
            window.location.href = '/login';
        });
    }


    // --- 5. INITIAL FUNCTION CALL ---
    // Call the function to get the count as soon as the page loads
    fetchSubmissionCount();
});