
import json
import json5
from vllm import SamplingParams

_EVAL_PROMPT_SYSTEM = """You are a fair and insightful judge with exceptional reasoning and analytical abilities. Your task is to evaluate a user's question, a generated response to that question, and an aspect that is important to the user. Based on this information, identify if the aspect is addressed in the generated response. Provide a clear and accurate assessment.

# your input:
    - question: the question asked by the user.
    - details: the detailed explanation of the question from the user.
    - response: a generated response to the user's question
    - aspect: the aspect that is important to the user, consisting of the following fields:
        - aspect: the title for the aspect.
        - reason: the reason that this aspect is important for the user.
        - evidence: the evidence from the user detailed explanation that the aspect extracted from.

# your output: Your output should be only a valid json object in ```json ``` block without any explanations that contains the following fields:
    - match_score: A score between 0 to 2 that indicates how well the generated response addresses this aspect, where: 0 means the response does not cover this aspect, 1 means the response somewhat covers this aspect, and 2 means the response covers this aspect very well.
"""

_EVAL_PROMPT_USER = """
question: {question}
details: {details}
response: {response}
aspect: {aspects}

Your output should be only a valid json object in ```json ``` block without any explanations.
"""

def parse_json(json_str):
    json_str = json_str.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(json_str, strict=False)
        return obj
    except:
        pass
    try:
        obj = json5.loads(json_str)
        return obj
    except:
        print(json_str)
        raise ValueError("Invalid json object")

def create_eval_prompt(question, details, response, aspect, tokenizer):
    aspect = f'-aspect: {aspect["aspect"]}\n    -reason: {aspect["reason"]}\n    -evidence: {aspect["evidence"]}'
    conversation = [
        {
            "role": "system",
            "content": _EVAL_PROMPT_SYSTEM
        },
        {
            "role": "user",
            "content": _EVAL_PROMPT_USER.format(question=question, details=details, response=response, aspects=aspect)
        }
    ]
    return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

def create_eval_prompts_all(queries, responses, details, aspects, tokenizer):
    prompts = []
    ids = []
    for i, (query, response, detail, aspect) in enumerate(zip(queries, responses, details, aspects)):
        for j, asp in enumerate(aspect):
            prompt = create_eval_prompt(query, detail, response, asp, tokenizer)
            prompts.append(prompt)
            ids.append(
                {
                    "q_id": i,
                    "a_id": j
                }
            )
    return ids, prompts
        

def evaluator(queries, responses, details, aspects, llm, max_retries=100):
    temperature = 0.0
    tokenizer = llm.get_tokenizer()
    retries = 0
    ids, prompts = create_eval_prompts_all(queries, responses, details, aspects, tokenizer)
    outputs_dict = {}
    while prompts:
        retries += 1
        sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=4096, logprobs=1)
        outputs = llm.generate(prompts, sampling_params)
        wrongs = []
        for id, prompt, output in zip(ids, prompts, outputs):
            if id['q_id'] not in outputs_dict:
                outputs_dict[id['q_id']] = {}
            try:
                obj = parse_json(output.outputs[0].text)
                score = obj['match_score']
                outputs_dict[id['q_id']][id['a_id']] = obj
            except Exception as e:
                print(e)
                if retries > max_retries:
                    outputs_dict[id['q_id']][id['a_id']] = {"match_score": 0}
                    continue
                wrongs.append((id, prompt))
        prompts = [prompt for id, prompt in wrongs]
        ids = [id for id, prompt in wrongs]
        if temperature < 1.0:
            temperature += 0.1
    scores = []
    for i, (query, aspect) in enumerate(zip(queries, aspects)):
        score_query = 0
        details = []
        for j, asp in enumerate(aspect):
            score_query += outputs_dict[i][j]['match_score']
            details.append({
                "aspect": asp['aspect'],
                "score": outputs_dict[i][j]['match_score']
            })
        scores.append({
            "id": i,
            "score": score_query / (len(aspect) * 2),
            "details": details
        })
    avg = {
        "score": sum([score['score'] for score in scores]) / len(scores),
        "per_question_scores": scores
    }
    return avg
