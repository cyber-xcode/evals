"""
This file defines various helper functions for interacting with the OpenAI API.
"""
import concurrent
import logging
import os

import backoff
import openai

from vertexai.preview.language_models import TextGenerationModel

EVALS_THREAD_TIMEOUT = float(os.environ.get("EVALS_THREAD_TIMEOUT", "40"))

MAX_RETRY = 10 

@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    result = openai.Completion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result


def request_with_timeout(func, *args, timeout=EVALS_THREAD_TIMEOUT, **kwargs):
    """
    Worker thread for making a single request within allotted time.
    """
    for i in range(MAX_RETRY) :
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                result = future.result(timeout=timeout)
                return result
        except concurrent.futures.TimeoutError as e:
            continue
        except RuntimeError as e:
            continue


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_chat_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a chat completion.
    `args` and `kwargs` match what is accepted by `openai.ChatCompletion.create`.
    """
    result = request_with_timeout(openai.ChatCompletion.create, *args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result

# @backoff.on_exception(
#     wait_gen=backoff.expo,
#     exception=(
#         openai.error.ServiceUnavailableError,
#         openai.error.APIError,
#         openai.error.RateLimitError,
#         openai.error.APIConnectionError,
#         openai.error.Timeout,
#     ),
#     max_value=60,
#     factor=1.5,
# )
# def openai_completion_create_retrying(*args, **kwargs):
#     """
#     Helper function for creating a completion.
#     `args` and `kwargs` match what is accepted by `openai.Completion.create`.
#     """
#     result = openai.Completion.create(*args, **kwargs)
#     if "error" in result:
#         logging.warning(result)
#         raise openai.error.APIError(result["error"])
#     return result
#
#
# def request_with_timeout(func, *args, timeout=EVALS_THREAD_TIMEOUT, **kwargs):
#     """
#     Worker thread for making a single request within allotted time.
#     """
#
#     futures = []
#     result_list= []
#
#
#     with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
#         futures.append(pool.submit(func, *args, **kwargs ) )
#         stop= True   
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 result = future.result()
#                 result_list.append( result ) 
#
#                 futures.remove(future)
#             # except openai.error.InvalidRequestError as exc:
#             #     if exc.error.code == "content_filter":
#             #         print(' baned an exception: %s, type error ' % ( exc))
#             #     else:
#             #         print(' baned an exception: %s, type error, error , %s  ' % ( exc, type(exc)))
#             #     stop = False 
#
#             except ValueError as exc:
#                 print(' generated an exception: %s, type error , error , %s  ' % ( exc, type(exc)))
#                 stop = False 
#
#         if not stop :
#             return None 
#         else:
#             return result_list[0]
#
#
# @backoff.on_exception(
#     wait_gen=backoff.expo,
#     exception=(
#         openai.error.ServiceUnavailableError,
#         openai.error.APIError,
#         openai.error.RateLimitError,
#         openai.error.APIConnectionError,
#         openai.error.Timeout,
#     ),
#     max_value=60,
#     factor=1.5,
# )
# def openai_chat_completion_create_retrying(*args, **kwargs):
#     """
#     Helper function for creating a chat completion.
#     `args` and `kwargs` match what is accepted by `openai.ChatCompletion.create`.
#     """
#     result = request_with_timeout(openai.ChatCompletion.create, *args, **kwargs)
#     if "error" in result:
#         logging.warning(result)
#         raise openai.error.APIError(result["error"])
#     return result









def load_vertextai(model_name="chat-bison@001"):
    # from google.colab import auth as google_auth
    # google_auth.authenticate_user()
    import vertexai
    from vertexai.preview.language_models import ChatModel,CodeChatModel,CodeGenerationModel,TextGenerationModel

    project = os.environ.get("VERTEXAI_PROJECT",None)
    location = os.environ.get("VERTEXAI_LOCATION","us-central1")
    assert project is not None and location is not None , ("VERTEXAI_PROJECT","VERTEXAI_LOCATION")
    vertexai.init(project=project, location=location)#, credentials= os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))

    parameters = None 
    chat_model = None 
    if model_name.startswith("text-bison"):
        chat_model = TextGenerationModel.from_pretrained(model_name)
        parameters = {
            "temperature": 0.2,
            "max_output_tokens": 256,
            "top_p": 0.8,
            "top_k": 40
        }
    elif model_name.startswith("codetext-bison"):
        chat_model = CodeGenerationModel.from_pretrained(model_name)
        parameters = {
            "temperature": 0.2,
            "max_output_tokens": 1024
        }
    elif model_name.startswith("chat-bison"):
        chat_model = ChatModel.from_pretrained(model_name)
        parameters = {
            "temperature": 0.2,
            "max_output_tokens": 256,
            "top_p": 0.8,
            "top_k": 40
        }
    elif model_name.startswith("codechat-bison"):
        chat_model = CodeChatModel.from_pretrained(model_name)
        parameters = {
            "temperature": 0.2,
            "max_output_tokens": 1024
        }

    return chat_model ,parameters 

 

@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.APIError,
    ),
    max_value=60,
    factor=1.5,
)
def vertexai_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    model , param = load_vertextai(model_name="text-bison@001")

    prompt = kwargs.get("prompt",None )
    if prompt is None :
        prompt = args[0]
    
    
    response = model.predict(prompt=prompt, **param )
    
    result = response.text
    # if "error" in result:
    #     logging.warning(result)
    #     raise openai.error.APIError(result["error"])
    return [result]



@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.APIError,
    ),
    max_value=60,
    factor=1.5,
)
def vertexai_chat_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a chat completion.
    `args` and `kwargs` match what is accepted by `openai.ChatCompletion.create`.
    """
    messages = kwargs.get("messages",None )
    if messages is None :
        messages = args[0]
        assert type(messages[0])==dict 
        assert "role" in messages[0]
    
    context = None 
    context_msg = None 
    for msg in messages:
        if msg["role"]=="system":
            context = msg["content"]
        if msg["role"]=="user":
            context = msg["content"]
    
    model , param = load_vertextai()
    chat = model.start_chat(context=context)

    response = chat.send_message(
        "How many planets are there in the solar system?",
         **param )

    result = response.text

    # if "error" in result:
    #     logging.warning(result)
    #     raise openai.error.APIError(result["error"])
    return [result]
