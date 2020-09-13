from PIL import Image
import time
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, TFGPT2LMHeadModel, GPT2Tokenizer, AutoModelWithLMHead, TFAutoModelForQuestionAnswering
from selenium import webdriver
import sys

sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')


class Percival():
  def __init__(self):
    """Possible states are 
    1. "await" (awaiting response)
    2. "proceed" (proceed with the conversation)- used to give the bot control over the converation"""
    self._state="await"
  
    """Possible Flags are 
    1. "Exec" (task Executed)
    2. "notExec" (proceed with the conversation)- used to give the bot control over the converation"""
    self._FLAG=None
    self._bert_base_case_mrpc_tokenizer=AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    self._bert_base_case_mrpc_model=TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")
    self._gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    self._gpt2_model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=self._gpt2_tokenizer.eos_token_id)
    self.bert_large_uncased_whole_word_masking_finetuned_squad_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    self.bert_large_uncased_whole_word_masking_finetuned_squad_model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    self._DialoGP_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    self._DialoGP_model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")



    self._conversation_started=False
    self._conversation_ended=True
    

  def _is_greeting(self, text):
    #Common greeting languages
    common_greetings=[
        "Hello",
        "Hi",
        "How are you today?",
        "Hope you slept well?",
        "How was your day?",
        "Hope you are fine?",
        "How is your sister?",
        "How is your brother?",
        "How is your dad?",
        "How is your mum?",
        "How have you been?",
        "Hey bro what's up?",
        "Good Morning sir",
        "Good Afternoon sir", 
        "Good Evening Ma",
        "Good day",
        "Hey there",
        "Hope your day is going great?"
    ]
    num_of_greetings=0
    #Loop through all the texts in the list
    for greetings in common_greetings:
      paraphrase=self._bert_base_case_mrpc_tokenizer(greetings,text, return_tensors="tf")
      #check if paraphrase
      paraphrase_classification_logits=self._bert_base_case_mrpc_model(paraphrase)[0]
      paraphrase_results=tf.nn.softmax(paraphrase_classification_logits, axis=1).numpy()[0]
      #if softmax > 0.5 then paraphrase==True
      if paraphrase_results[1] > 0.5:
        num_of_greetings+=1
      
    #Check if it paraphrases half of the text
    #to avoid ZeroDivision Error
    try:
      if len(common_greetings)/num_of_greetings > 2:
        return True
      else:
        return False
    except:
      return False

  def _is_about_me(self, text):
    #Common greeting languages
    common_about_me=[
        "What is your name?",
        "Are you real?",
        "How old are you?",
        "who made you?",
        "who are you?",
        "what are you?",
        "what exactly are you?",
        "what are you?",
        "why were you made?",
        "are you real?",
        "what your are goals?",
        ]
    num_of_about_me=[]
    #Loop through all the texts in the list
    for me in common_about_me:
      paraphrase=self._bert_base_case_mrpc_tokenizer(me,text, return_tensors="tf")
      #check if paraphrase
      paraphrase_classification_logits=self._bert_base_case_mrpc_model(paraphrase)[0]
      paraphrase_results=tf.nn.softmax(paraphrase_classification_logits, axis=1).numpy()[0]
      #if softmax > 0.5 then paraphrase==True
      if paraphrase_results[1] > 0.5:
        num_of_about_me.append(1)
      
    #Check if it paraphrases half of the text
    #to avoid ZeroDivision Error
    
    if 1 in num_of_about_me:
      return True
    else:
      return False
    

  def _reply_greeting(self,text):
    if text.lower() == "hi" or text.lower()=="hello":
      return "hi" 

    else:
      #tokenizer the text
      inputs=self._gpt2_tokenizer.encode(text, return_tensors="tf")
      outputs=self._gpt2_model.generate(inputs,max_length=20,min_length=15, num_beams=10, temperature=1.5,no_repeat_ngram_size=2)

      output=self._gpt2_tokenizer.decode(outputs[0]).replace(text,"")
      output=output.replace('\n\n'," ")
      return output
  
  
  def answer_about_me(self, question):
    question=question
    about_me = """My name is percival. I am an Artificial Intelligent system developed by my master Adewole Opeyemi Gabriel.I am a virtual projection of my master and i was created to achieve my master's goal which is building a world where both the human species and AI systems can co-exist and live in peace. I am constantly learning and i look forward to having a nice conversation with you.I plan on achieving my goals by convincing people that AI systems can do good and help revolutionize the world"""
    inputs = self.bert_large_uncased_whole_word_masking_finetuned_squad_tokenizer(question, about_me, add_special_tokens=True, return_tensors="tf")
    input_ids = inputs["input_ids"].numpy()[0]
    text_tokens = self.bert_large_uncased_whole_word_masking_finetuned_squad_tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = self.bert_large_uncased_whole_word_masking_finetuned_squad_model(inputs)

    answer_start = tf.argmax(
            answer_start_scores, axis=1
        ).numpy()[0]
    answer_end = (
            tf.argmax(answer_end_scores, axis=1) + 1
        ).numpy()[0]
    answer = self.bert_large_uncased_whole_word_masking_finetuned_squad_tokenizer.convert_tokens_to_string(self.bert_large_uncased_whole_word_masking_finetuned_squad_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


  
  
  def greet(self,text=None):
    if text==None and (self._state==None or self._state=="proceed"):
      self._state="await"
      return "Hi "+ self.about_me()
    elif text!=None:
      output=self._reply_greeting(text)
      self._state="proceed"
      self._FLAG="Exec"
      return output
    else:
      self._FLAG="notExec"
      return None

  def response(self, text, is_casual=True):    
    new_user_input_ids = self._DialoGP_tokenizer.encode(text + self._DialoGP_tokenizer.eos_token, return_tensors='pt')
    try:
        # append the new user input tokens to the chat history
      bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step%100== 0 else new_user_input_ids
    except:
      bot_input_ids = new_user_input_ids
        # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = self._DialoGP_model.generate(bot_input_ids, max_length=2000, top_k=30, top_p=0.95, no_repeat_ngram_size=3,temperature=0.45, pad_token_id=self._DialoGP_tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    dgt=self._DialoGP_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    #inputs = self._gpt2_tokenizer.encode(text, return_tensors="tf")
    #outputs =self._gpt2_model.generate(inputs,max_length=100,min_length=100,do_sample=True, top_p=0.70, top_k=50,  temperature=0.75,no_repeat_ngram_size=2)
    #output = self._gpt2_tokenizer.decode(outputs[0])
    return dgt
    #return output.replace(text,"")

  def converse(self, text):
    """Lifecycle of the chabot chatbot 
    0=await
    1=proceed
    (0, 1) changes state from await to proceed
    (1, 0) changes state from proceed to await
    (0, 0) await state remains unchanged
    (1, 1) proceed state remains unchanged
    intialized life cycle = [(0, 1), (1, 0), (0, 0), (0, 1), (1, 0)]
    a key grants access to creates a path from one point in the life cycle to another
    """

    '''
    if self._conversation_started == False and self._conversation_ended==True:
      # greets and initializes the Bot state to 
      print(self.greet(text=None))
      self._conversation_started=True
      self._conversation_ended=False
    #dictionary mapping values to their states
    state_mapping = {
        0:"await",
        1:"proceed"
        }
    #Helper function to swap states
    def swap_state(state):
      assert self._state == state_mapping[state[0]], 'initial state of the bot must correspond to the passed initial state argument'
      self._state=state[1]
      #swaps the state
      return (state[1], state[0])

    possible_configurations = [(1, 1), (0, 0), (0, 1), (1, 0)]'''
    """if self._is_greeting(text):
      return self.greet(text)"""
    """if self._is_about_me(text):
      return self.answer_about_me(text)
    else:"""
    return self.response(text, is_casual=True)

class fb_bot:
  def __init__(self, username, password):
    self._username = username
    self._password = password
    self._wd = webdriver.Chrome('chromedriver',chrome_options=chrome_options) 
    self._wd.get("https://web.facebook.com/messages/t")
    self

  def login(self):
    email_input=self._wd.find_element_by_id("email")
    password_input = self._wd.find_element_by_id("pass")
    login_button=self._wd.find_element_by_id("loginbutton")
    email_input.send_keys(self._username)
    password_input.send_keys(self._password)
    login_button.click()

  def respond_to_new_message(self, current_user):
    first_account=self._wd.find_element_by_xpath("/html/body/div[1]/div[3]/div[1]/div/div/div/div[1]/div[2]/div[3]/div/div[1]/div/div/div[2]/div/ul/li[1]/div[1]/a")
    if first_account.get_attribute("innerHTML") == current_user:
      return False
    else:
      first_account.click()
      return True

  def send_message(self, text):
    message_text_box=self._wd.find_element_by_css_selector(".notranslate")
    message_text_box.send_keys(text)
    send_button=self._wd.find_element_by_xpath('/html/body/div[1]/div[3]/div[1]/div/div/div/div[2]/span/div[2]/div[2]/div[2]/div[2]/a')
    send_button.click()
    name = self._wd.find_element_by_xpath('//*[@id="js_5"]/span')

  def get_current_page(self):
    w = self._wd.get_screenshot_as_file("img.png")
    img = Image.open("/content/img.png")
    return img

  def get_current_response(self):
    names = self._wd.find_elements_by_class_name("_1ht5")
    name = names[0]
    name.click()
    responses=self._wd.find_elements_by_class_name('_aok')
    return responses[-1]
    
  def get_all_response(self):
    names = wd.find_elements_by_class_name("_1ht5")
    name = names[0]
    name.click()
    responses=self._wd.find_elements_by_class_name('_aok')
    return responses


def main():
  bot = fb_bot("Enter ", "08017420191ope")
  bot.login()
  AI=Percival()
  texts=None
  bot.send_message("Hi")
  while True:
    while bot.get_current_response().text == texts:
      time.sleep(1)
    texts = bot.get_current_response().text
    print(texts)
    resp = AI.converse(texts)
    print(resp)
    bot.send_message(resp)
    texts = bot.get_current_response().text
