from manual import LLM_response, LLM_response_multi, extract_answer
from newmanual import qwitho_prompt_few, qwitho_prompt_few_chinese, Deductive_zero_prompt, Lextract_prompt_few, Simple_prompt_few, Muti2one_zero_prompt, Deductive_noFOL_zero_prompt, Deductive_noDeductive_zero_prompt


# def few_qwitho(Question, Option, model):
def few_qwitho(Question, Option):
    # proposition = model.invoke(qwitho_prompt_few.format(Question=Question, Option=Option)).content
    proposition = LLM_response(qwitho_prompt_few_chinese.format(Question=Question, Option=Option))
    # print("!!!!!!!:", qwitho_prompt_few_chinese.format(Question=Question, Option=Option))
    return proposition

def few_qwitho_eng(Question, Option):
    # proposition = model.invoke(qwitho_prompt_few.format(Question=Question, Option=Option)).content
    proposition = LLM_response(qwitho_prompt_few.format(Question=Question, Option=Option))
    # print("!!!!!!!:", qwitho_prompt_few_chinese.format(Question=Question, Option=Option))
    return proposition

# def Logic_qwitho(Question, Option, model):
def Logic_qwitho(Question, Option):
    proposition = few_qwitho(Question, Option)
    return proposition

def Logic_qwitho_eng(Question, Option):
    proposition = few_qwitho_eng(Question, Option)
    return proposition

def Logic_deductive(Context, Proposition):
    content = f"Context: {Context}\nProposition: {Proposition}\n{Deductive_zero_prompt}"
    reasoning = LLM_response(content) 
    return reasoning

def Logic_Lextract(Context):
    Lextract = LLM_response(Lextract_prompt_few.format(Natural_language_text = Context))
    return Lextract

def Logic_simple(Context):
    simple = LLM_response(Simple_prompt_few.format(Origin_context = Context))
    return simple

def Logic_m2one(eng_context, eng_ques, option_list, analyse_list, eng_options):
    content = f"Context: {eng_context}\nQusetion: {eng_ques}\nOptions: {option_list}\nReasoning processes provided by the student: {analyse_list}\n{Muti2one_zero_prompt}"
    response = LLM_response_multi(content)
    answer = [0, 0, 0, 0]
    for i in range(5):
        response_2 = response[i]['message']['content']
        while True:
            try: 
                print("Muti to One:", response_2)
                content_4 = f"You now have an analysis: {response_2} and several options: {eng_options}. Please carefully read the analyse and directly output the number of the option that the analyse ultimately considers correct from the options, using letters A/B/C/D, Please direct output the A/B/C/D:"
                response_3 = LLM_response(content_4)
                answer_0 = int(extract_answer(response_3))
                answer[answer_0] += 1
                break
            except:
                print("try again")
    
    max_number = max(answer)
    
    # 找到该数第一次出现的位置
    max_position = answer.index(max_number)
    
    return max_position

def Logic_deductive_noFOL(Context, Proposition):
    content = f"Context: {Context}\nProposition: {Proposition}\n{Deductive_noFOL_zero_prompt}"
    reasoning = LLM_response(content) 
    return reasoning

def Logic_nodeductive(Context):
    content = f"Context: {Context}\n{Deductive_noDeductive_zero_prompt}"
    reasoning = LLM_response(content) 
    return reasoning