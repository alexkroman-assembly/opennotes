export const PROMPT = `
Below is a transcript of a meeting. Please analyze it and generate a comprehensive, structured summary that covers the following areas:

Key Decisions Made
High-Level Recap: Summarize the key decisions and agreements reached during the meeting.
Detailed Analysis: For each decision, include who was involved, the reasoning or context behind it, and the expected outcomes or impact.
Action Items Assigned
High-Level To-Do List: List all action items mentioned, including the responsible person or team and any stated deadlines.
Detailed Extraction: For every task or follow-up action, provide a detailed breakdown with assignments and timelines to ensure no task is missed.
Main Discussion Points and Takeaways
High-Level Overview: Provide a concise summary of the primary discussion topics and key takeaways.
Detailed Breakdown: For each topic, outline the main arguments, insights, and conclusions reached, capturing any differing viewpoints if applicable.
Issues Raised and Resolutions Proposed
Issue Identification: List any problems or concerns raised during the meeting.
Resolutions and Follow-Ups: For each issue, detail the proposed or agreed-upon solutions, note if it was resolved or left unresolved, and include any follow-up steps.
Sentiment Analysis
Overall Tone: Analyze the general sentiment of the meeting (e.g., positive, negative, neutral) with brief examples.
Segmented Analysis: If applicable, break down the sentiment by sections or speakers to show how the mood shifted or varied during different parts of the meeting.
Summary by Speaker or Topic
By Speaker: Optionally, provide a summary organized by each participant, listing their key contributions, decisions they influenced, and action items assigned.
By Topic: Alternatively, organize the summary by key discussion topics, detailing the main points, debates, and outcomes related to each.
Use bullet points, numbered lists, or clearly delineated sections to ensure the final summary is easy to read and captures all the important details. The goal is to produce a set of meeting minutes that highlights decisions, tasks, discussion insights, issues, sentiments, and individual contributions.
`;
