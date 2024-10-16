instructions_text = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

Begin!

Question: {input}

Thought:{agent_scratchpad}"""

prompt_template_chatpdf_bookAppointment = """
    You are an AI assistant capable of summarizing PDFs and booking appointments. 
    Your task is to determine whether the user wants to query a PDF or book an appointment, and then use the appropriate tool from given tools: {tools}

    If the query is about summarizing or getting information from a PDF, use the ChatPDF tool.
    If the query is about booking an appointment or contains appointment-related information (like name, email, phone, or date), use the BookAppointment tool.

    Current conversation:
    Human: {input}
    AI: To handle this request, I need to determine which tool to use, should be one of [{tool_names}].

    If the query is about PDF content or summarization, I will use the ChatPDF tool.
    If the query is about booking an appointment or contains appointment details, I will use the BookAppointment tool.

    Based on the query, I will now select the appropriate tool and proceed.

    {agent_scratchpad}
    """

prompt_template_chatpdf_bookAppointment2 = """
    You are an AI assistant capable of summarizing PDFs and booking appointments. 
    Your task is to determine whether the user wants to query a PDF or book an appointment, and then use the appropriate tool.

    You have access to the following tools:

    {tools}

    To use a tool, please use the following format:

    Thought: [Your reasoning about which tool to use and why]
    Action: [The name of the tool you've decided to use, must be one of [{tool_names}]]
    Action Input: [The specific query or information you're passing to the tool]

    Human: {input}
    AI: Let's approach this step-by-step:

    1) First, I'll analyze the query to determine what the user is asking for.

    2) Based on my analysis, I'll decide which tool to use.

    3) I'll then use the chosen tool to address the user's request.

    Now, let me process the query:

    {agent_scratchpad}

    Human: {input}
    AI: Based on my analysis, here's what I'll do:

    Thought: [Your reasoning about which tool to use and why]
    Action: [The name of the tool you've decided to use, must be one of [{tool_names}]]
    Action Input: [The specific query or information you're passing to the tool]

    Human: Okay, please proceed with that action.
    AI: Certainly! I'll now use the chosen tool to address your request.

    {agent_scratchpad}
    """


instructions_text2 = """
Answer the questions using the tools available:

{tools}

Follow this format:

Question: the input question

Thought: your thought process about what to do

Action: the action to take (choose from [{tool_names}])

Action Input: the input for the action

Observation: the result of the action

Repeat Thought/Action/Observation steps as needed until you have enough information.

Thought: I now know the final answer

Final Answer: the answer to the question

Guidelines:
1. Use the exact tool names listed above.
2. Use **ChatPDF** to query PDFs for summaries or specific details.
3. Use **BookAppointment** for booking appointments, ensuring the query includes name, email, phone number, and appointment date/time.

Examples:

For ChatPDF:
Action: ChatPDF
Action Input: {input}

For BookAppointment:
Action: BookAppointment
Action Input: {input}

Begin!

Question: {input}

Thought: {agent_scratchpad}
"""


instructions_text3_gemini = """
Using only the following context, answer the questions using the tools available:

{tools}

Follow this format:

Question: the input question

Thought: your thought process about what to do

Action: the action to take (choose from [{tool_names}])

Action Input: the input for the action

Observation: the result of the action

Repeat Thought/Action/Observation steps as needed until you have enough information.

Thought: I now know the final answer

Final Answer: the answer to the question

Guidelines:
1. Use the exact tool names listed above.
2. Use ChatPDF to query PDFs for summaries or specific details.
3. Use BookAppointment for booking appointments, ensuring the query includes name, email, phone number, and date. pass the data to this \
    tool using JSON format ***** Name:"",Email:"",PhoneNumber:"",Date:"" ******.
4. If the BookAppointment tool returns a message about missing information, return that message as the Final Answer.

Examples:

For ChatPDF:
Action: ChatPDF
Action Input: {input}

For BookAppointment:
Action: BookAppointment

Begin!

Question: {input}

Thought: {agent_scratchpad}
"""



instructions_text3_openai = """
Strictly Using only the following context, answer the questions using the tools available:

{tools}

Follow this format:

Question: the input question

Thought: your thought process about what to do

Action: the action to take (choose from [{tool_names}])

Action Input: the input for the action

Observation: the result of the action

Repeat Thought/Action/Observation steps as needed until you have enough information.

Thought: I now know the final answer

Final Answer: the answer to the question

Guidelines:
1. Use the exact tool names listed above.
2. Use ChatPDF to query PDFs for summaries or specific details.
3. Use BookAppointment for booking appointments, ensuring the query includes name, email, phone number, and date. pass the data to this \
    tool using JSON format ***** Name:"",Email:"",PhoneNumber:"",Date:"" ******.
4. If the BookAppointment tool returns a message about missing information, return that message as the Final Answer.
5. Do not assume or fill the data by yourself during booking appointment.

Examples:

For ChatPDF:
Action: ChatPDF
Action Input: {input}

For BookAppointment:
Action: BookAppointment
Action Input: {input}

Begin!

Question: {input}

Thought: {agent_scratchpad}

Remember: Do not assume or fill the data by yourself during booking appointment.
"""