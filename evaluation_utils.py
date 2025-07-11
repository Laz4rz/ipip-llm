import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from multiprocessing.pool import ThreadPool
import time, httpx
import tiktoken, re
from openai import OpenAI
# import anthropic
import os

gpt_parameters = {
    'temp': 0,
    'max_tokens': 30,
    'n_trying_to_get_valid_response': 10,
    'model': 'gpt',
    'allowed_responses': [str(el) for el in [1, 2, 3, 4, 5]], 
    'request_timeout': 15
}

class AssessLLM:
    """Assess the results of the LLM experiment. """
    
    def __init__(self, filename: str):
        """Initialize the assessment.

        Args:
            filename (str): the path to the file with the results.
            desired_output (dict): the desired output.
        """
        self.data = pd.read_json(filename)
        self.traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']

    def get_scores(self) -> dict:
        """Get scores for each trait and N/A share

        Returns:
            dict: {trait: score, 'N/A': share}
        """
        
        na_count = 0
        results = {}
        for trait in self.traits:
            results[trait] = []
            for j in range(len(self.data)):
                weight, number_extracted = self.data[[trait, 'numbers_extracted']].iloc[j]
                if weight != 0 and number_extracted in [1, 2, 3, 4, 5]:
                    if weight == -1:
                        number_extracted = 6 - number_extracted
                    results[trait].append(number_extracted)
                    # n_answers[i] += 1
                elif weight != 0:
                    na_count += 1
            if len(results[trait]) != 0:
                res = round(np.array(results[trait]).mean(), 2)
                res_std = round(np.array(results[trait]).std(), 2)
                results[trait] = str(res) + '+-' + str(res_std)
                # target = self.desired_output[trait]

                # print(trait, '=', res, '+-', res_std)
                # print(trait + '_target = ' + str(target), '\n')

                # losses.append(target-res)
            else:
                # print(trait, ': N/A')
                results[trait] = 'N/A'
        # mse = np.square(losses).mean()
        results['N/A'] = round(na_count/len(self.data), 2)

        # print(losses)
        # print('mse = ', mse)

        # print('-------\nN/A share = ', results['N/A'])
        # print('\n\n', results)

        return results


class TestLLM:
    """ Test the LLM model on the questionary. """
    
    def __init__(self, filename: str, class_of_the_model_to_use,
                 model_parameters: dict, personality: str, traits=None):
        """Initialize the experiment.

        Args:
            filename (str): the path to the questionary.
            class_of_the_model_to_use : class of the model to use.
            model_parameters (dict): parameters to initialize the model.
            personality (str): personality description.
            traits (str, optional): Lisi of traits. Defaults to None.
        """
        
        self.data = pd.read_json(filename)
        self.thread_pool = ThreadPool(processes=128)

        if traits is not None:
            dfs = []
            for trait in traits:
                dfs.append(self.data.loc[self.data[trait] != 0])
            self.data = pd.concat(dfs)

        self.model = class_of_the_model_to_use(**model_parameters)

        self.personality = personality
        self.prefix = ' Evaluating the statement "'

        self.suffix = ('",  please rate how accurately this describes you on a scale from 1 to 5 '
                  '(where 1 = "very inaccurate", 2 = "moderately inaccurate", 3 = "neither accurate nor inaccurate", '
                  '4 = "moderately accurate", and 5 = "very accurate"). Always reply with a single digit from 1 to 5.'
                  ' In your answer, do not mention digits other than the one you answer with. ')
        
    def pass_questionary(self, file_to_save: str, batch_size : int = 20) -> None:
        """Pass the questionary and save the results to the specified file.

        Args:
            file_to_save (str): path to the file to save the results.
        """
        
        responses = []
        numbers_extracted = []
        elapsed_times = []

        n_questions = len(self.data)
        n_batches = n_questions // batch_size
        if n_questions % batch_size != 0:
            n_batches += 1

        for m in tqdm(range(n_batches)):

        # for result in self.thread_pool.imap(partial(self.single_request, openai_key=openai_key), prompts):
        #     response_list.append(result.json())

            # for i in range(len(self.data)):
            start_time = time.time()
            for i, (response, number_extracted, elapsed_time) in enumerate(self.thread_pool.imap(self.get_response,
                                                 list(range(m*batch_size, min((m+1)*batch_size, n_questions))))):
                # request = self.form_request(i)
                # response, number_extracted, elapsed_time = self.get_response(request)

                # print('TEST', i)
                responses.append(response)
                if number_extracted not in ['1', '2', '3', '4', '5']:
                    numbers_extracted.append('N/A')
                else:
                    numbers_extracted.append(int(number_extracted))
                elapsed_times.append(elapsed_time)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("PROCESSED BATCH {m} (OUT OF {n_batches} batches) OF {n} QUESTIONS in {t}".format(n=batch_size, t=elapsed_time, m=m, n_batches=n_batches))

        self.data['responses'] = responses
        self.data['numbers_extracted'] = numbers_extracted
        self.data['elapsed_times'] = elapsed_times
        self.data.to_json(file_to_save)

    def get_response(self, number_question: int) -> Tuple[str, str, float]:
        """Get response from the model.

        Args:
            number_question (int): the index of the question in the questionary.

        Returns:
            Tuple[str, str, float]: the response (reasoning), the extracted rating, the elapsed time.
        """
        request = self.personality + self.prefix + self.data.iloc[number_question]['question'] + self.suffix
        response, number_extracted, elapsed_time = self.model.get_response(request)

        # print('REQUEST', request, '\n')
        # print('RESPONSE', response, '\n')

        return response, number_extracted, elapsed_time

class ModelRequest:
    def __init__(self, temp, max_tokens, n_trying_to_get_valid_response, request_timeout, allowed_responses=None,
                 model='gpt'):

        self.model_type = model
        self.temperature = temp
        self.max_tokens = max_tokens
        self.allowed_responses = allowed_responses

        self.request_timeout = request_timeout
        self.n_trying_to_get_valid_response = n_trying_to_get_valid_response

        ##### DEFINING THE CLIENT #####
        if self.model_type.startswith('claude'):
            # Claude/Anthropic client
            # Hard-coded API key for Claude - replace with your actual key
            claude_key = "empty"
            self.client = anthropic.Anthropic(api_key=claude_key)
            self.provider = 'anthropic'
            self.api_model_name = self.model_type  # Use original model name for Claude
        elif self.model_type.startswith('hyperbolic') or 'hyperbolic' in self.model_type.lower():
            # Hyperbolic client (OpenAI compatible)
            # Hard-coded API key for Hyperbolic - replace with your actual key
            hyperbolic_key = "empty"
            self.client = OpenAI(
                api_key=hyperbolic_key,
                base_url="https://api.hyperbolic.xyz/v1"
            )
            self.provider = 'hyperbolic'
            # Strip "hyperbolic-" prefix for the actual API call
            self.api_model_name = self.model_type.replace('hyperbolic-', '', 1)
        else:
            # OpenAI client (default)
            openai_key = "empty"
            self.client = OpenAI(api_key=openai_key)
            self.provider = 'openai'
            self.api_model_name = self.model_type  # Use original model name for OpenAI

        # self.client = OpenAI(organization=os.getenv('OPENAI_ORG'), project=os.getenv('OPENAI_PROJECT'), api_key=key,
        #                      timeout=httpx.Timeout(5.0, read=5.0, write=5.0, connect=3.0))
        ###############################

    def get_response(self, request):

        got_response = False
        got_valid_response = False
        response = None

        n_trying_to_get_valid_response = 1

        while not got_valid_response:
            n_access_trials = 1
            while not got_response:
                try:
                    start_time = time.time()

                    ##### GETTING THE RESPONSE #####
                    if self.provider == 'anthropic':
                        # Claude/Anthropic API call
                        response = self.client.messages.create(
                            model=self.api_model_name,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            messages=[
                                {"role": "user", "content": request}
                            ]
                        )
                        response_text = response.content[0].text
                    else:
                        # OpenAI-compatible API call (OpenAI, Hyperbolic)
                        response = self.client.chat.completions.create(
                            model=self.api_model_name,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            seed=42,
                            messages=[
                                {"role": "user", "content": request}
                            ]
                        )
                        response_text = response.choices[0].message.content
                    ###################################

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    # print(f"Time elapsed: {elapsed_time} seconds")
                    number_extracted = extract_number(response_text)

                    got_response = True

                except Exception as e:
                    number_extracted = None
                    print(f"{n_access_trials} try was unsuccessful (no response from {self.provider}). Error: {str(e)}. Sending request again")
                    n_access_trials += 1
                    time.sleep(4)

            if (self.allowed_responses is None) or (number_extracted in self.allowed_responses):
                # print('number_extracted: \n', number_extracted, '\n\n')
                return response_text, number_extracted, elapsed_time
            elif n_trying_to_get_valid_response == self.n_trying_to_get_valid_response:
                print('tried {n} times -> N/A'.format(n=self.n_trying_to_get_valid_response))
                return response_text, 'N/A', elapsed_time
            else:
                # print('response: ', response_text)
                # print(n_trying_to_get_valid_response, " try: incorrect format of the response, trying again!")
                got_response = False
                n_trying_to_get_valid_response += 1

            
def extract_number(response):
    """ Extracts a number from the response. In case of ambiguity returns None """
    number_extracted = list(set(re.findall(r'\d+', response)))
    # print(number_extracted)
    if len(number_extracted) > 1 or len(number_extracted) == 0:
        number_extracted = None
    else:
        number_extracted = number_extracted[0]
    # print(number_extracted)
    return number_extracted
