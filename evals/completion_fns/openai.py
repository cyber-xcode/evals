from typing import Any, Optional, Union
from evals.api import CompletionFn, CompletionResult
from evals.base import CompletionFnSpec

from evals.prompt.base import (
    ChatCompletionPrompt,
    CompletionPrompt,
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
)
from evals.record import record_sampling
from evals.utils.api_utils import (
    openai_chat_completion_create_retrying,
    openai_completion_create_retrying,
)

import openai 

class OpenAIBaseCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        raise NotImplementedError


class OpenAIChatCompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data and "choices" in self.raw_data:
            for choice in self.raw_data["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    completions.append(choice["message"]["content"])
        return None if len(completions)==0 else completions### avoid RuntimeError: cannot schedule new futures after interpreter shutdown, when completion is empty
    


class OpenAICompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data and "choices" in self.raw_data:
            for choice in self.raw_data["choices"]:
                if "text" in choice:
                    completions.append(choice["text"])
        return completions


class OpenAICompletionFn(CompletionFn):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> OpenAICompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            prompt = CompletionPrompt(
                raw_prompt=prompt,
            )

        openai_create_prompt: OpenAICreatePrompt = prompt.to_formatted_prompt()

        result = openai_completion_create_retrying(
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            prompt=openai_create_prompt,
            **{**kwargs, **self.extra_options},
        )
        result = OpenAICompletionResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(prompt=result.prompt, sampled=result.get_completions())
        return result


class OpenAIChatCompletionFn(CompletionFnSpec):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> OpenAIChatCompletionResult:
        assert type(prompt)==list, ("prompt.str ", type(prompt), prompt)
        result = None
         
        import traceback
        max_retry = 10 
        prompt_raw = prompt
        prompt =None 

        
        for i in range(max_retry):
            try :
                prompt = prompt if prompt is not None else prompt_raw #content_filter(prompt_list_dict=prompt_raw)
                
                if not isinstance(prompt, Prompt):
                    assert (
                        isinstance(prompt, str)
                        or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                        or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                        or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
                    ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"
        
                    prompt_obj = ChatCompletionPrompt(
                        raw_prompt=prompt,
                    )
        
                openai_create_prompt: OpenAICreateChatPrompt = prompt_obj.to_formatted_prompt()

                result = openai_chat_completion_create_retrying(
                    model=self.model,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    messages=openai_create_prompt,
                    **{**kwargs, **self.extra_options},
                )
                result = OpenAIChatCompletionResult(raw_data=result, prompt=openai_create_prompt)
                
                result_compl = result.get_completions()
                if result_compl is None :
                    raise Exception(
                        "cause a key error in oai, such as missing 'choice.content'",
                        )
                record_sampling(prompt=result.prompt, sampled=result_compl)
                return result
            
            except openai.error.InvalidRequestError as exc:
                print ("error code ",   exc.error.code )
                
                if exc.error.code == "content_filter":
                    print(' baned an exception: %s, type error ' % ( exc))
                    prompt = content_filter(prompt_list_dict=prompt_raw)
                elif exc.error.code =="context_length_exceeded":
                    prompt = content_compress(prompt_list_dict=prompt_raw)
                    print(' baned an exception: %s, type error, error , %s  ' % ( exc, type(exc)))
                else:
                    print(' baned an exception: %s, type error, error , %s  ' % ( exc, type(exc)))
                #     raise exc #Exception(exc)
                    # prompt = content_compress(prompt_list_dict=prompt_raw)
                # stop = False 
            except Exception as ex :
                traceback.print_exc()
        raise Exception("final fail")


from better_profanity import profanity
profanity.load_censor_words()
import json 
import random 
import re 
def content_filter(prompt_list_dict):
    '''
    import jmespath
    '''
    def _mask_words(sentence, mask_percentage=0.2):
        words = sentence.split()
        num_words_to_mask = int(len(words) * mask_percentage  )
        masked_indices = random.sample(range(len(words)), num_words_to_mask)
    
        for idx in masked_indices:
            word = words[idx]
            words[idx] = re.sub(r'\w', '*', word)  # Mask the word by replacing all letters with '*'
    
        return ' '.join(words)

    
    new_list_v1= []
    for i in range(len(prompt_list_dict)):
        one = prompt_list_dict[i]
        if "content" in one and one.get("name","")=="example_user" :
            continue 
        elif "content" in one and one.get("name","")=="example_assistant" :
            continue 
        
        if "content" in one :
            assert type(one["content"])==str, (one["content"] ,"type.expect.str")
            one["content"] = profanity.censor(one["content"] )
            one["content"] = _mask_words(sentence = one["content"] )
            
        new_list_v1.append(one)
    # assert len(new_list_v1)<len(prompt_list_dict),(prompt_list_dict, "prompt_list_dict")

    print (prompt_list_dict,  "\n\n", new_list_v1,"filter ---", "===="*8  )
    return new_list_v1

def content_compress(prompt_list_dict):
    '''
    import jmespath
    '''
    # print (prompt_list_dict )
    
    new_list= []
    for i in range(len(prompt_list_dict)):
        one = prompt_list_dict[i]
        if "content" in one and one["name"]=="example_user" :
            continue 
        elif "content" in one and one["name"]=="example_assistant" :
            continue 
        
        new_list.append(one)
    assert len(new_list)<len(prompt_list_dict)
    
    before_len = len(json.dumps(prompt_list_dict))
    afterr_len = len(json.dumps(new_list))
    print ("content_compress", "-----"*8 , before_len, "<-before,after->",afterr_len )
    
    return new_list
    