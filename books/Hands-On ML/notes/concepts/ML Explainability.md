# ML Explainability
A **white box model** is one whose decisions are **intuitive and easy to interpret**. #decision-tree  models are a good example: they boil down to simple rules you could even apply by hand, and you can trace exactly why any given prediction was made.

In contrast, **black box models** like #random-forest and #neural-networks make great predictions and let you inspect every computation they performed — yet it's usually **hard to explain in simple terms *why*** a prediction came out the way it did. If a neural network says a person appears in a picture, it's hard to tell what drove the call: their eyes, their mouth, their nose, or even the couch they were sitting on.

The field of #interpretable-ml aims to build ML systems that can explain their decisions in a way humans can understand.