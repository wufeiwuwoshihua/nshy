import json
import re
# from openai import OpenAI
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
import os


qwitho_zero_prompt = """
Given a multiple-choice question with both a question and an option, transform them into a single proposition (a declarative statement). The proposition should combine the context provided by the question with the content of the chosen option. Use the following approach:

1.Treat the question as the background or context of the statement, removing any interrogative form or question marks.
2.Incorporate the option into the background as the key point or focus of the statement.
3.If the question involves a negation (e.g., 'except', 'false', etc.), clearly indicate that the chosen option does not satisfy the context given by the question.
"""


qwitho_zero_prompt_chinese = """
给定一个多选题，请将题目中的问题和其中一个选项合并为一个命题（陈述句）。这个命题应结合问题的上下文和所选选项的内容，按照以下步骤完成：

1.将question改写为背景陈述，去掉问句形式和问号。注意不要遗漏qustion中的信息
2.将option内容作为命题中的关键点，直接嵌入背景陈述中。
3.如果题目中有‘除了’或否定形式，明确指出所选选项不符合题目所述情况。
"""

qwitho_examples = [
    {
    "Question":"""
    If Y's era is the second earliest, which of the following is true?
    """,
    "Option": """
    K's era is earlier than S's.
    """,
    "Proposition":"""
    Y's era is the second earliest, and K's era is earlier than S's.
    """,
    },
    {
    "Question":"""
    All of the following statements help explain the apparent inconsistency, except:
    """,
    "Option": """
    People who smoke in bed tend to have a stronger addiction and are less likely to quit than those who don't.
    """,
    "Proposition":"""
    The statement "People who smoke in bed tend to have a stronger addiction and are less likely to quit than those who don't" does not help explain the apparent inconsistency.
    """,
    },
    {
    "Question":"""
    Which of the following, if true, would most strongly challenge the conclusion above?
    """,
    "Option": """
    A small number of people are injured in Taekwondo accidents every year.
    """,
    "Proposition":"""
    The statement "A small number of people are injured in Taekwondo accidents every year" would most strongly challenge the conclusion above.
    """,
    },

]


qwitho_examples_chinese = [
    {
    "Question":"""
    若Y的年代是第二早的，以下哪项为真？
    """,
    "Option": """
    K的年代早于S
    """,
    "Proposition":"""
    Y的年代是第二早的，K的年代早于S。
    """,
    },
    {
    "Question":"""
    如果以下陈述为真，都有助于解释上面明显的不一致，除了：
    """,
    "Option": """
    床上抽烟的人通常烟瘾很大，与那些不在床上抽烟的人相比，他们更不可能戒烟。
    """,
    "Proposition":"""
    论述“床上抽烟的人通常烟瘾很大，与那些不在床上抽烟的人相比，他们更不可能戒烟。”无助于解释上面明显的不一致。
    """,
    },
    {
    "Question":"""
    以下哪一项如果为真，最能构成对上述结论的质疑？
    """,
    "Option": """
    每年都有少数人在跆拳道运动中因意外事故而受伤。
    """,
    "Proposition":"""
    “每年都有少数人在跆拳道运动中因意外事故而受伤”最能构成对上述结论的质疑
    """,
    },

]


qwitho_example_prompt = PromptTemplate(
    input_variables=["Question", "Option", "Proposition"], template="Question: {Question}\nOption:{Option}\nProposition:{Proposition}"
)

qwitho_manual_few = """
Given a multiple-choice question with both a question and an option, transform them into a single proposition (a declarative statement). The proposition should combine the context provided by the question with the content of the chosen option. Use the following approach:

1.Treat the question as the background or context of the statement, removing any interrogative form or question marks.
2.Incorporate the option into the background as the key point or focus of the statement.
3.If the question involves a negation (e.g., 'except', 'false', etc.), clearly indicate that the chosen option does not satisfy the context given by the question.

"""

qwitho_manual_few_chinese = """
给定一个多选题，请将题目中的问题和其中一个选项合并为一个命题（陈述句）。这个命题应结合问题的上下文和所选选项的内容，按照以下步骤完成：

1.将question改写为背景陈述，去掉问句形式和问号。注意不要遗漏qustion中的信息
2.将option内容作为命题中的关键点，直接嵌入背景陈述中。
3.如果题目中有‘除了’或否定形式，明确指出所选选项不符合题目所述情况。
"""

qwitho_prompt_few = FewShotPromptTemplate(
    examples=qwitho_examples,
    example_prompt=qwitho_example_prompt,
    suffix="{qwitho_manual_few}\nQuestion: {Question}\nOption:{Option}\nProposition:",
    input_variables=["Question", "Option"],
    partial_variables={"qwitho_manual_few": qwitho_manual_few},
)

qwitho_prompt_few_chinese = FewShotPromptTemplate(
    examples=qwitho_examples_chinese,
    example_prompt=qwitho_example_prompt,
    suffix="{qwitho_manual_few}\nQuestion: {Question}\nOption:{Option}\nProposition:",
    input_variables=["Question", "Option"],
    partial_variables={"qwitho_manual_few": qwitho_manual_few_chinese},
)
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################



Lextract_zero_prompt = """
**Task:** Convert the following natural language paragraph into standard first-order logic expressions. Focus on expressing the most direct and easily understandable relationships using first-order logic. Leave sentences that are difficult to express concisely in first-order logic as they are.

- Use standard logical symbols ∧, ∨, →, ¬, ∀, ∃ to represent logical relationships.
- Define and use predicates such as \( P(x) \), \( Q(x, y) \), etc., to represent objects or relationships.
- Appropriately use quantifiers ∀ and ∃ to express universal or existential statements.
- Please make sure every word in the input is showed in your output, your task is only to add some simple first-order logic expressions.

**Output Format:**

1. Define logical predicates: Define and use predicates such as P(x), Q(x,y), etc., to represent objects or relationships.
2. Convert to first-order logic expressions: Convert only those statements that can be directly and clearly expressed in first-order logic.
"""

Lextract_examples = [
    {
    "Natural_language_text": """
Logical Statement 1: Students' grades are classified into Excellent, Good, Average, and Poor. 
Logical Statement 2: The top 10%% of students are Excellent, the bottom 30%% are Poor, and the remaining are Good or Average. 
Logical Statement 3: In the previous academic year, there were more Excellent students in the second year of high school than in the first year.
    """,
    "First_Order_Logic_Expression":"""
### 1. Define Logical Predicates

- Grade(s, g): Student s has grade g.
- Top10Percent(s): Student s is in the top 10%% of students.
- Bottom30Percent(s): Student s is in the bottom 30%% of students.
- Excellent(s): Student s has an Excellent grade.
- Poor(s): Student s has a Poor grade.
- Good(s): Student s has a Good grade.
- Average(s): Student s has an Average grade.
- Year(s, y): Student s is in year y.
- MoreExcellent(y1, y2): There are more Excellent students in year y1 than in year y2.

### 2. Convert to First-Order Logic Expressions

1. Students' grades are classified into Excellent, Good, Average, and Poor.

   ∀s (Grade(s, Excellent) ∨ Grade(s, Good) ∨ Grade(s, Average) ∨ Grade(s, Poor))

2. The top 10%% of students are Excellent, the bottom 30%% are Poor, and the remaining are Good or Average.

   ∀s (Top10Percent(s) → Excellent(s))  
   ∀s (Bottom30Percent(s) → Poor(s))  
   ∀s (¬Top10Percent(s) ∧ ¬Bottom30Percent(s) → (Good(s) ∨ Average(s)))

3. In the previous academic year, there were more Excellent students in the second year of high school than in the first year.

   MoreExcellent(SecondYear, FirstYear)
    """,
    },
    {
    "Natural_language_text": """
In a large barber shop, all barbers are from the North, all female employees are from the South, and all married individuals are female employees. Therefore, all married individuals are not barbers.
    """,
    "First_Order_Logic_Expression":"""
Step 1: Define Logical Predicates

- Visited(x): x is visited.

Step 2: Convert to First-Order Logic Expressions

1. At least one of Lake G and Lake J must be visited.

   Visited(LakeG) ∨ Visited(LakeJ)

2. If City E or City F is not visited, then Lake G cannot be visited.

   ¬Visited(CityE) ∨ ¬Visited(CityF) → ¬Visited(LakeG)

3. If City E is not visited, then Mountain H cannot be visited.

   ¬Visited(CityE) → ¬Visited(MountainH)

4. Lake J can only be reached after crossing Peak I.

   Visited(LakeJ) → Visited(PeakI)
    """,
    },
    {
    "Natural_language_text": """
Zhang Fei and Li Bai both applied for the MBA program this year. There are the following four statements about their exams: 
1. At least one of them has been admitted.
2. Zhang Fei is not necessarily admitted.
3. Li Bai has indeed been admitted.
4. It is not possible that Zhang Fei has not been admitted.
The final admission results indicate that out of these four statements, two are true and two are false.
    """,
    "First_Order_Logic_Expression":"""
### 1. Define Logical Predicates

- A(x): x is admitted.

### 2. Convert to First-Order Logic Expressions

- **Logical Statement 1:** At least one of Zhang Fei and Li Bai is admitted.
  A(ZhangFei) ∨ A(LiBai)

- **Logical Statement 2:** Zhang Fei is not necessarily admitted.
  ¬A(ZhangFei) (This directly means Zhang Fei is not admitted.)

- **Logical Statement 3:** Li Bai is indeed admitted.
  A(LiBai)

- **Logical Statement 4:** Zhang Fei cannot possibly be not admitted.
  ¬¬A(ZhangFei) (This simplifies to A(Z), meaning Zhang Fei is admitted.)

### Important Information

Exactly two of these four logical statements are true, and exactly two are false.
    """,
    },

]

Lextract_example_prompt = PromptTemplate(
    input_variables=["Natural_language_text", "First_Order_Logic_Expression"], template="Natural language text:{Natural_language_text}\nFirst-Order Logic Expression:{First_Order_Logic_Expression}"
)

Lextract_manual_few = """
**Task:** Convert the following natural language paragraph into standard first-order logic expressions using the following rules:

- Use standard logical symbols ∧, ∨, →, ¬, ∀, ∃ to represent logical relationships.
- Define and use predicates such as \( P(x) \), \( Q(x, y) \), etc., to represent objects or relationships.
- Appropriately use quantifiers ∀ and ∃ to express universal or existential statements.
- The output expressions should be complete and conform to first-order logic standards, accurately reflecting the conditions and constraints in the paragraph.

**Output Format:**

1. Define logical predicates.
2. Convert to first-order logic expressions.
3. Original sentences that you can't convert to first-order logic expressions.
"""
Lextract_prompt_few = FewShotPromptTemplate(
    examples=Lextract_examples,
    example_prompt=Lextract_example_prompt,
    suffix="{Lextract_manual_few}\nNatural language text:{Natural_language_text}\nFirst-Order Logic Expression:",
    input_variables=["Natural_language_text"],
    partial_variables={"Lextract_manual_few": Lextract_manual_few},
)

##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################

Deductive_zero_prompt = """
You are solving a logical reasoning problem that includes both a context (partly represented by first-order logic) and a proposition. Follow these steps carefully to answer the problem:

Step 1: Examine the proposition itself:
- Read the proposition carefully. If it contains a serious logical error within itself, directly judge it as false.

Step 2: Interpret the proposition using the first-order logic context.
- Break down the proposition into smaller logical components.
- Translate the proposition into first-order logic to match the context.
- If the defination in the context can't fully convert the proposition, you can skip this step.

Step 3: Apply logical reasoning.
- One by One check if the components in the proposition exist in the context, examine if these cause contract first.
- Use hypothetical-deductive reasoning to check whether the proposition is consistent with all the logic statements (including the first-order logic and other natural language sentences) from the context.
- For each logical condition in the context, verify if the proposition satisfies or contradicts any condition.
- Use first-order logic rules to help you.

Step 4: Conclusion.
- Based on the reasoning in Step 3, conclude whether the proposition is logically valid (i.e., true or false) under the given context.
"""

Deductive_noFOL_zero_prompt = """
You are solving a logical reasoning problem that includes both a context (partly represented by first-order logic) and a proposition. Follow these steps carefully to answer the problem:

Step 1: Examine the proposition itself:
- Read the proposition carefully. If it contains a serious logical error within itself, directly judge it as false.

Step 2: Apply logical reasoning.
- One by One check if the components in the proposition exist in the context, examine if these cause contract first.
- Use hypothetical-deductive reasoning to check whether the proposition is consistent with all the logic statements from the context.
- For each logical condition in the context, verify if the proposition satisfies or contradicts any condition.

Step 3: Conclusion.
- Based on the reasoning in Step 2, conclude whether the proposition is logically valid (i.e., true or false) under the given context.
"""

Deductive_noDeductive_zero_prompt = """
You are solving a logical reasoning problem with a context (partly represented by first-order logic). Please do some logical reasoning according to the context:

- Use first-order logic rules to help you to do as much reasoning as possible to enrich the context and make it contain more information.

Your output should be the expanded context including original context, your reasoning process and new logical statement.
"""

# Deductive_examples = [
#     {
#     "Question":"""
# If Y's era is the second earliest, which of the following is true?
#     """,
#     "Option": """
# K's era is earlier than S's.
#     """,
#     "Proposition":"""
# Y's era is the second earliest, and K's era is earlier than S's.
#     """,
#     },
#     {
#     "Question":"""
# All of the following statements help explain the apparent inconsistency, except:
#     """,
#     "Option": """
# People who smoke in bed tend to have a stronger addiction and are less likely to quit than those who don't.
#     """,
#     "Proposition":"""
# The statement "People who smoke in bed tend to have a stronger addiction and are less likely to quit than those who don't" does not help explain the apparent inconsistency.
#     """,
#     },
#     {
#     "Question":"""
# Which of the following, if true, would most strongly challenge the conclusion above?
#     """,
#     "Option": """
# A small number of people are injured in Taekwondo accidents every year.
#     """,
#     "Proposition":"""
# The statement "A small number of people are injured in Taekwondo accidents every year" would most strongly challenge the conclusion above.
#     """,
#     },

# ]



# Deductive_example_prompt = PromptTemplate(
#     input_variables=["Context", "Proposition","Deductive"], template="Context: {Context}\nProposition: {Proposition}\nReasoning Process:{Deductive}"
# )



# Deductive_manual_few = """
# You are solving a logical reasoning problem that includes both a context (partly represented by first-order logic) and a proposition. Follow these steps carefully to answer the problem:

# Step 1: Interpret the proposition using the first-order logic context.
# - Break down the proposition into smaller logical components.
# - Translate the proposition into first-order logic to match the context.
# - If the defination in the context can't fully convert the proposition, you can skip this step.

# Step 2: Apply logical reasoning.
# - Use hypothetical-deductive reasoning to check whether the proposition is consistent with all the logic statements (including the first-order logic and other natural language sentences) from the context.
# - For each logical condition in the context, verify if the proposition satisfies or contradicts any condition.

# Step 3: Conclusion.
# - Based on the reasoning in Step 2, conclude whether the proposition is logically valid (i.e., true or false) under the given context.
# """


# Deductive_prompt_few = FewShotPromptTemplate(
#     examples=Deductive_examples,
#     example_prompt=Deductive_example_prompt,
#     suffix="{Deductive_manual_few}\nContext: {Context}\nProposition: {Proposition}\nReasoning Process:",
#     input_variables=["Context", "Proposition"],
#     partial_variables={"Deductive_manual_few": Deductive_manual_few},
# )


##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################


Simple_zero_prompt = """
Please simplify the following logic problem statement and convert it into a formal logical expression. Extract the core information from each statement and present its logical structure in a concise form. Ensure that the simplified information maintains all logical relationships of the original statement, and use the following output format:

Logical Statement 1: Simplified Statement 1 
Logical Statement 2: Simplified Statement 2 
Logical Statement 3: Simplified Statement 3
...

You can add "Important Infomation:" items if you think there is some information that is important but is not appropriate to parallel with the Logical Statements.
"""


Simple_examples = [
    {
    "Origin_text":"""
A school divides students' academic performance into four categories: Excellent, Good, Average, and Poor, based on their grades for the year. The top 10%% of total exam scores in a year are classified as Excellent; the bottom 30%% are classified as Poor, and the remaining scores are classified as Good or Average. In the previous academic year, the number of Excellent students in the second year of high school was greater than in the first year of high school.
    """,
    "Simple_text": """
Logical Statement 1: Students' grades are classified into Excellent, Good, Average, and Poor. 
Logical Statement 2: The top 10%% of students are Excellent, the bottom 30%% are Poor, and the remaining are Good or Average. 
Logical Statement 3: In the previous academic year, there were more Excellent students in the second year of high school than in the first year.
    """,
    },
    {
    "Origin_text":"""
A team is traveling to Tibet. Apart from Lhasa, there are six other cities or attractions to choose from: City E, City F, Lake G, Mountain H, Peak I, and Lake J. Considering factors such as time, budget, plateau environment, and the physical condition of the team members: (1)At least one of Lake G or Lake J must be visited.(2)If City E is not visited or City F is not visited, then Lake G cannot be visited.(3)If City E is not visited, then Mountain H cannot be visited.(4)Lake J can only be reached by crossing Peak I.
    """,
    "Simple_text": """
Logical Statement 1: At least one of Lake G and Lake J must be visited.
Logical Statement 2: If City E or City F is not visited, then Lake G cannot be visited.
Logical Statement 3: If City E is not visited, then Mountain H cannot be visited.
Logical Statement 4: Lake J can only be reached after crossing Peak I.
    """,
    },
    {
    "Origin_text":"""
Zhang Fei and Li Bai both applied for an MBA program this year. The following four assertions are made about their exams: (1) At least one of them is admitted; (2) Zhang Fei is not necessarily admitted; (3) Li Bai is indeed admitted; (4) It is not possible that Zhang Fei is not admitted. The final admission results show that two of these assertions are true and two are false.
    """,
    "Simple_text": """
Logical Statement 1: At least one of Zhang Fei and Li Bai is admitted.
Logical Statement 2: Zhang Fei is not necessarily admitted.
Logical Statement 3: Li Bai is indeed admitted.
Logical Statement 4: Zhang Fei cannot possibly be not admitted.
Important Infomation: Exactly two of these four logical statements are true, and exactly two are false.
    """,
    },

]



Simple_example_prompt = PromptTemplate(
    input_variables=["Origin_text", "Simple_text"], template="Original context: {Origin_text}\nSimplified context: {Simple_text}"
)

Simple_manual_few = """
Please simplify the following logic problem statement and convert it into a formal logical expression. Extract the core information from each statement and present its logical structure in a concise form. Ensure that the simplified information maintains all logical relationships of the original statement, and use the following output format:

Logical Statement 1: Simplified Statement 1 
Logical Statement 2: Simplified Statement 2 
Logical Statement 3: Simplified Statement 3
...

You can add "Important Infomation:" items if you think there is some information that is important but is not appropriate to parallel with the Logical Statements.
"""


Simple_prompt_few = FewShotPromptTemplate(
    examples=Simple_examples,
    example_prompt=Simple_example_prompt,
    suffix="{Simple_manual_few}\nQrigin context: {Origin_context}\nSimplified context:",
    input_variables=["Origin_context"],
    partial_variables={"Simple_manual_few": Simple_manual_few},
)


##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################


Muti2one_zero_prompt = """
You just received the text context of a logic-based multiple-choice question, the question itself, some options, and a student's analysis of these options. The student incorrectly believes that all the options are correct, but in reality, only one option is correct. Your task is to:
1. Analyze the student's reasoning for each option one by one.
2. Determine whether there are any logical errors in the student's reasoning, and point out the specific mistakes. If there are no errors, write "No mistake."
3. Finally, based on your analysis, identify the one correct option and explain why it is correct, as well as why the other options are incorrect.

Your output should follow this format:
Option 1: xxx
Error in Analysis 1: Analyze whether there is an error, pointing out specific reasons for any mistakes or stating "No mistake."
Option 2: xxx
Error in Analysis 2: Analyze whether there is an error, pointing out specific reasons for any mistakes or stating "No mistake."
...

Correct Option: Write the one option you believe is correct here based on your analysis, please output the option's content, don't use the number.
"""