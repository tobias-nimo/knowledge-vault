There are specific techniques that can improve LLM outputs by leveraging their instruction-following abilities...
## Zero shot
A zero-shot prompt is the simplest type of prompt. It only provides a description of a task
and some text for the LLM to get started with. This input could be anything: a question, a
start of a story, or instructions. The name zero-shot stands for ’**no examples**’.

An example of zero-shot prompting for **classifying movie reviews**...
- LLM output configuration:
	```json
	{
		"model": "gemini-pro",
		"temperature": 0.1,
		"token_limit": 5,
		"top_k": "None",
		"top_p": 1
	}
	```
- Prompt:
	```txt
	Classify movie reviews as POSITIVE, NEUTRAL or NEGATIVE. 
	
	Review: "Her" is a disturbing study revealing the direction humanity is headed if AI is allowed to keep evolving, unchecked. I wish there were more movies like this masterpiece. 
	
	Sentiment:
	```
- Output:
	```txt
	POSITIVE
	```

> When zero-shot doesn’t work, you can provide demonstrations or examples in the prompt, which leads to one-shot and few-shot prompting.
## One-shot & few-shot
When creating prompts for AI models, it is helpful to **provide examples**:
- A **one-shot** prompt, provides a single example, hence the name one-shot.
- A **few-shot** prompt provides multiple examples to the model.

These examples can help the model understand what you are asking for. Examples are especially useful when you want to steer the model to a certain **output structure or pattern**.

As a general **rule of thumb**, you should use at least 3 to 5 examples for few-shot prompting. However, you may need to use more examples for more complex tasks.

An example of few-shot prompting for **parsing pizza orders to JSON**:
- LLM output configuration:
	```json
	{
		"model": "gemini-pro",
		"temperature": 0.1,
		"token_limit": 250,
		"top_k": "None",
		"top_p": 1
	}
	```
- Prompt:
	```txt
	Parse a customer's pizza order into valid JSON:
	
	EXAMPLE: 
	I want a small pizza with cheese, tomato sauce, and pepperoni. 
	JSON Response:
	{
	"size": "small",
	"type": "normal",
	"ingredients": [["cheese", "tomato sauce", "peperoni"]]
	}
	
	EXAMPLE:
	Can I get a large pizza with tomato sauce, basil and mozzarella
	JSON Response:
	{
	"size": "large",
	"type": "normal",
	"ingredients": [["tomato sauce", "bazel", "mozzarella"]]
	}
	
	Now, I would like a large pizza, with the first half cheese and
	mozzarella. And the other tomato sauce, ham and pineapple.
	JSON Response:
	```
- Output:
	```json
	{
	"size": "large",
	"type": "half-half",
	"ingredients": [
			["cheese", "mozzarella"],
			["tomato sauce", "ham", "pineapple"]
		]
	}
	```

> When you choose examples for your prompt, use examples that are **relevant** to the task you want to perform. The examples should be **diverse**, of **high quality**, and **well written**. One small mistake can confuse the model and will result in undesired output.

> If you are trying to generate output that is robust to a variety of inputs, then it is important to include **edge cases** in your examples. Edge cases are inputs that are unusual or unexpected, but that the model should still be able to handle.
## System, contextual and role prompting 
System, contextual and role prompting are all techniques used to guide how LLMs generate text, but they focus on different aspects:
- **System prompting** sets the overall context and purpose for the language model. It defines the ‘big picture’ of what the model should be doing, like translating a language, classifying a review etc.
- **Contextual prompting** provides specific details or background information relevant to the current conversation or task. It helps the model to understand the nuances of what’s being asked and tailor the response accordingly.
- **Role prompting** assigns a specific character or identity for the language model to adopt. This helps the model generate responses that are consistent with the assigned role and its associated knowledge and behavior.

There can be considerable overlap between system, contextual, and role prompting. E.g. a prompt that assigns a role to the system, can also have a context. However, each type of prompt serves a slightly different primary purpose:
- System prompt: Defines the model’s fundamental capabilities and overarching purpose.
- Contextual prompt: Provides immediate, task-specific information to guide the response. It’s highly specific to the current task or input, which is dynamic.
- Role prompt: Frames the model’s output style and voice. It adds a layer of specificity and personality.




