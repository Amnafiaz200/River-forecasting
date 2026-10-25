"""AI-use disclosure required at the end of the report (prompt, output, edits)."""
from rl_helpers import *


def build():
    s = [P("AI-use disclosure", H1)]
    s.append(P("<b>Tool used:</b> Claude Code (Anthropic), model Claude Sonnet 5.5, run inside the project folder. "
               "Below are the prompt given to the tool, a description of what it produced, and the edits made. "
               "<b>Student note:</b> the student must read this section and the whole report, then add their own edits in the last row before submitting."))
    s.append(P("Prompt (verbatim, as given)", H3))
    prompt = ("First, inspect every file I uploaded carefully. There is a PDF that contains the complete assignment instructions, requirements, steps, "
              "and submission details. Do not start solving immediately. First, study the assignment completely and understand: what the assignment is asking me to do; "
              "every requirement and instruction in the PDF; what files, code, data, or other materials are provided; which files I need to use; what steps need to be "
              "completed and in what order; what calculations, analysis, coding, screenshots, outputs, explanations, or evidence are required; any formatting, naming, or submission requirements; "
              "what files must be run; what final files must be submitted; any grading criteria, restrictions, or important details that could cause marks to be lost. "
              "After fully understanding the assignment, create a clear internal plan and then complete the entire assignment according to those instructions. "
              "Important requirements: 1. Follow the PDF instructions exactly. Do not invent requirements or skip anything. 2. Use the provided files as the main source of truth. "
              "3. Complete every required question, task, calculation, code section, analysis, and output. 4. If coding is required, inspect the existing project/files before making changes. "
              "5. Do not randomly rewrite or modify working code. Change only what is required by the assignment. 6. Make sure the code actually runs and produces the required results. "
              "7. Validate the final results against the assignment instructions. 8. Check for missing sections, incorrect calculations, incomplete answers, formatting problems, or required evidence before finishing. "
              "9. Keep explanations clear, simple, and appropriate for a student assignment. 10. Do not leave placeholders such as “TODO,” “add screenshot,” “complete later,” or unfinished sections "
              "unless the assignment requires information that genuinely cannot be obtained from the provided materials. 11. Do not stop after partially completing the work. "
              "Finish the assignment in one complete pass as far as the provided materials allow. Before giving me the final answer, perform a final checklist against every instruction in the PDF "
              "and make sure nothing has been missed. At the end, give me a clear summary with: A. What you completed. B. How to run it. C. What results I should expect. D. What to submit. E. Final verification.")
    s.append(P(prompt, SMALL))
    s.append(P("Output of the AI tool", H3))
    s += bullets([
        "Question 1: the full architecture, assumptions, equations, operations and uncertainty text, and the two diagrams (source: <font name='DVM'>Question1/make_diagram.py</font>, <font name='DVM'>report/q1_text.py</font>).",
        "Question 2: all code (<font name='DVM'>Question2/mm.py, checks.py, flow_fig1.py, train.py, analysis.py, select_d.py, run_all.sh</font>), the training runs, all tables and figures, the answers Q2.1–Q2.17, including the two predictions written before the experiments "
        "(Q2.5, Q2.11), which the tool wrote and committed to the repository before running the experiments they refer to.",
        "The PDF builder (<font name='DVM'>report/build_report.py</font>) that takes every number from the JSON files in <font name='DVM'>Question2/results/</font>.",
    ])
    s.append(P("Edits made", H3))
    s += bullets([
        "Edits by the AI tool during the session: fixed the edge count (6, not 7) in the Q1 diagram and text after noticing the mismatch; no results were edited by hand.",
        "<b>Edits by the student:</b> (to be filled in by the student after reviewing the report — list any changes made to text, code or numbers).",
    ])
    return s
