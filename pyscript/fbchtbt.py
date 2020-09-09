import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, TFGPT2LMHeadModel, GPT2Tokenizer

class ChatBot():
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
    self._gpt2_model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
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
        "Tell me about yourself",
        "Are you real?",
        "How old are you?",
        "who made you?",
        "who are you?",
        "what are exactly are you?",
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
    

  def _reply_greeting(text):
    if text.lower() == "hi" or text.lower()=="hello":
      return "hi" 

    else:
      #tokenizer the text
      inputs=self._gpt2_tokenizer.encode(text, return_tensors="tf")
      outputs=self._gpt2_model.generate(inputs,max_length=20,min_length=15, num_beams=10, temperature=1.5,no_repeat_ngram_size=2)

      output=self._gpt2_tokenizer.decode(outputs[0]).replace(text,"")
      output=output.replace('\n\n'," ")
      return output
  
  
  def about_me(self):
    about_me = """I am an Artificial Intelligent system developed by my master Adewole Opeyemi Gabriel.I am a virtual projection of my master and i was created to achieve my master's goal which is building a world where both the human species and AI systems can co-exist and live in peace. I am constantly learning and i look forward to having a nice conversation with you."""
    return about_me.replace("\n","")

  
  
  def greet(self,text=None):
    if text==None and (self._state==None or self._state=="proceed"):
      self._state="await"
      return "Hi "+ self.about_me()
    elif text!=None:
      output=_reply_greeting(text)
      self._state="proceed"
      self._FLAG="Exec"
      return output
    else:
      self._FLAG="notExec"
      return None
  
  def response(self, text):
    inputs = self._gpt2_tokenizer.encode(text, return_tensors="tf")
    outputs =self._gpt2_model.generate(inputs,max_length=100,min_length=100,do_sample=True, top_p=0.70, top_k=50,  temperature=0.75,no_repeat_ngram_size=2)
    output = tokenizer.decode(outputs[0])
    return output.replace(text,"")

  def converse(self):
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

    possible_configurations = [(1, 1), (0, 0), (0, 1), (1, 0)]
    while not self._conversation_ended:
      text=input("Let's chat: ")
      if text.lower == "stop":
        self._conversation_started=False
        self._conversation_ended=True
        break
      else:
        if self._is_greeting(text):
          print(self.greet(text))
        elif self._is_about_me(text):
          print(self.about_me())
        else:
          print(self.response(text))
 
