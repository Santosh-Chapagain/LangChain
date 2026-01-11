from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
import os
from langchain_core.runnables import RunnableParallel

load_dotenv()


model1 = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model2 = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template='Generate short and simple note from the following text \n {text}',
    input_variables=['text']

)

prompt2 = PromptTemplate(
    template= 'Generate 5 short question answer from the following text \n {text}' ,
    input_variables= ['text']

)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quizes into a single document \n notes => {notes} and quizes => {quiz} ' ,
    input_variables=['notes' , 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model2 | parser ,
    'quiz': prompt2 | model1 | parser

})

merge_chain = prompt3 | model1 | parser 
chain = parallel_chain | merge_chain 

text = """
Support Vector Machine (SVM)
1. What is SVM?

A Support Vector Machine (SVM) is a supervised machine learning algorithm used for:

Classification

Regression

Outlier detection

SVM finds the best decision boundary (a line in 2D, plane in 3D, hyperplane in higher dimensions) that separates data points of different classes with the maximum possible margin.

The goal is not just to separate classes, but to separate them as confidently as possible.

2. Geometric Intuition

Suppose you have two classes:

Class +1 (positive)

Class −1 (negative)

There are infinitely many lines that can separate them.

SVM chooses the one that:

Maximizes the distance from the closest points of both classes.

These closest points are called Support Vectors — they support or define the position of the boundary.

The margin is the distance between:

The closest positive point

The closest negative point

A larger margin means:

Better generalization

More robustness to noise

3. Mathematical Representation

Let the training data be:

{
(
𝑥
1
,
𝑦
1
)
,
(
𝑥
2
,
𝑦
2
)
,
.
.
.
,
(
𝑥
𝑛
,
𝑦
𝑛
)
}
{(x
1
	​

,y
1
	​

),(x
2
	​

,y
2
	​

),...,(x
n
	​

,y
n
	​

)}

where

𝑥
𝑖
∈
𝑅
𝑑
,
𝑦
𝑖
∈
{
+
1
,
−
1
}
x
i
	​

∈R
d
,y
i
	​

∈{+1,−1}

We want to find a hyperplane:

𝑤
⋅
𝑥
+
𝑏
=
0
w⋅x+b=0

where

𝑤
w = weight vector

𝑏
b = bias

Classification rule:

𝑦
=
sign
(
𝑤
⋅
𝑥
+
𝑏
)
y=sign(w⋅x+b)
4. Margin

The distance of a point 
𝑥
x from the hyperplane is:

∣
𝑤
⋅
𝑥
+
𝑏
∣
∥
𝑤
∥
∥w∥
∣w⋅x+b∣
	​


The margin (distance between the two class boundaries) is:

2
∥
𝑤
∥
∥w∥
2
	​


So maximizing margin is equivalent to minimizing:

∥
𝑤
∥
∥w∥
5. Hard-Margin SVM (Perfectly Separable Data)

We impose constraints:

𝑦
𝑖
(
𝑤
⋅
𝑥
𝑖
+
𝑏
)
≥
1
y
i
	​

(w⋅x
i
	​

+b)≥1

Optimization problem:

min
⁡
𝑤
,
𝑏
1
2
∥
𝑤
∥
2
w,b
min
	​

2
1
	​

∥w∥
2

subject to

𝑦
𝑖
(
𝑤
⋅
𝑥
𝑖
+
𝑏
)
≥
1
y
i
	​

(w⋅x
i
	​

+b)≥1

This finds the maximum-margin hyperplane.

6. Soft-Margin SVM (Real Data)

Real data has noise, so we allow errors using slack variables 
𝜉
𝑖
ξ
i
	​

.

Constraints:

𝑦
𝑖
(
𝑤
⋅
𝑥
𝑖
+
𝑏
)
≥
1
−
𝜉
𝑖
y
i
	​

(w⋅x
i
	​

+b)≥1−ξ
i
	​


Objective:

min
⁡
𝑤
,
𝑏
,
𝜉
(
1
2
∥
𝑤
∥
2
+
𝐶
∑
𝜉
𝑖
)
w,b,ξ
min
	​

(
2
1
	​

∥w∥
2
+C∑ξ
i
	​

)

Here:

𝐶
C controls trade-off between margin size and misclassification

Large 
𝐶
C → less error allowed

Small 
𝐶
C → more tolerance

7. Dual Formulation

Using Lagrange multipliers 
𝛼
𝑖
α
i
	​

, we convert to:

max
⁡
𝛼
∑
𝛼
𝑖
−
1
2
∑
𝛼
𝑖
𝛼
𝑗
𝑦
𝑖
𝑦
𝑗
(
𝑥
𝑖
⋅
𝑥
𝑗
)
α
max
	​

∑α
i
	​

−
2
1
	​

∑α
i
	​

α
j
	​

y
i
	​

y
j
	​

(x
i
	​

⋅x
j
	​

)

subject to:

∑
𝛼
𝑖
𝑦
𝑖
=
0
,
𝛼
𝑖
≥
0
∑α
i
	​

y
i
	​

=0,α
i
	​

≥0

Only data points with 
𝛼
𝑖
>
0
α
i
	​

>0 become support vectors.

8. Kernel Trick (Non-linear SVM)

When data is not linearly separable, SVM maps it to a higher-dimensional space:

𝑥
→
𝜙
(
𝑥
)
x→ϕ(x)

Instead of computing 
𝜙
(
𝑥
)
ϕ(x) explicitly, we use a kernel:

𝐾
(
𝑥
𝑖
,
𝑥
𝑗
)
=
𝜙
(
𝑥
𝑖
)
⋅
𝜙
(
𝑥
𝑗
)
K(x
i
	​

,x
j
	​

)=ϕ(x
i
	​

)⋅ϕ(x
j
	​

)

Common kernels:

Kernel	Formula
Linear	
𝑥
𝑖
⋅
𝑥
𝑗
x
i
	​

⋅x
j
	​


Polynomial	
(
𝑥
𝑖
⋅
𝑥
𝑗
+
1
)
𝑑
(x
i
	​

⋅x
j
	​

+1)
d

RBF (Gaussian)	
𝑒
−
𝛾
∥
𝑥
𝑖
−
𝑥
𝑗
∥
2
e
−γ∥x
i
	​

−x
j
	​

∥
2

Sigmoid	
tanh
⁡
(
𝑥
𝑖
⋅
𝑥
𝑗
)
tanh(x
i
	​

⋅x
j
	​

)

This allows SVM to create non-linear decision boundaries.

9. Why SVM is Powerful

SVM:

Works well in high-dimensional spaces

Avoids overfitting via maximum margin

Uses only support vectors → efficient memory use

Handles non-linear data using kernels

10. Summary

Support Vector Machine finds the hyperplane that:

Separates classes

Maximizes margin

Uses only critical data points (support vectors)

Can be extended to non-linear data via kernels

It is one of the most mathematically elegant and powerful classifiers in machine learning.

"""

result = chain.invoke({'text': text})
print(result)