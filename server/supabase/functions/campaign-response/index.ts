import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

// CORS headers for security
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Authenticate the request
    const authHeader = req.headers.get('Authorization')
    if (authHeader !== `Bearer ${Deno.env.get('FUNCTION_SECRET')}`) {
      return new Response("Unauthorized", { status: 401, headers: corsHeaders })
    }

    const { campaign_id, student_email } = await req.json()
    if (!campaign_id || !student_email) {
      return new Response("Missing campaign_id or student_email", { status: 400, headers: corsHeaders })
    }

    // Create an admin client to interact with the database
    const supabaseAdmin = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    // Find the student's UUID from their email
    const { data: student } = await supabaseAdmin
      .from('students')
      .select('id')
      .eq('email', student_email)
      .single()

    if (!student) {
      return new Response("Student not found", { status: 404, headers: corsHeaders })
    }

    // Insert the response into the campaign_responses table
    const { error: insertError } = await supabaseAdmin
      .from('campaign_responses')
      .insert({ campaign_id: campaign_id, student_id: student.id })

    if (insertError) { throw insertError }

    return new Response(JSON.stringify({ message: "Response recorded" }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    })
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 500,
    })
  }
})