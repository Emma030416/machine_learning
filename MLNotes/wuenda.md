# Introduction
machine learning algorithms:<br>
supervised learning <-> unsupervised learning
![描述](./img/wuenda01.png)

## 📌 supervised learning
input-> output label<br>
learn from being given "right answers"

applications using supervised learning:

| Input (X)         | Output (Y)             | Application         |
| ----------------- | ---------------------- | ------------------- |
| email             | spam? (0/1)            | **spam filtering**      |
| audio             | text transcripts       | speech recognition  |
| English           | Spanish                | machine translation |
| ad, user info     | click? (0/1)           | online advertising  |
| image, radar info | position of other cars | self-driving car    |
| image of phone    | defect? (0/1)          | visual inspection   |

examples:<br>
1. housing price prediction:<br>
whether to fit a straight line, a curve or another function to the data
![描述](./img/wuenda02.png)
**regression**: predict numbers / continuous valued output

2. breast cancer detection:<br>
![描述](./img/wuenda03.png)
**classification**: predict categories / discrete valued output

the examples above only provide one input or feature, in fact, more than one feature also works<br>
![描述](./img/wuenda04.png)
find a boundary line<br>
we use **SVM(Support Vector Machine)** when we have infinite numbers of features

## 📌 unsupervised learning
find the structure or pattern by itself in unlabeled data

**clustering**: group data into different clusters

examples:<br>
1. google news:<br>
find the acticles with similar words and group them into the same cluster
this is used in Recommender Systems, recommend related articles in the same cluster

2. Cocktail Party Algorithm:<br>
for knowing only<br>
a typical problem of BSS(Blind Source Separation), solving by ICA or Sparse Coding)

# Linear Regression with One Variable
## 📌 model
example: housing price prediction

m: number of training examples<br>
x: input(feature)<br>
y: output(target/label)<br>
(x, y): one training example<br>
(x⁽ⁱ⁾, y⁽ⁱ⁾): the ith training example<br>
h: function of the model(f(x))

so the model is like below:
![描述](./img/wuenda05.png)

## 📌 cost function 
after setting up our model, we need to choose the reasonable parameters: θo and θ1

what means reasonable? --- minimize the modeling error between predicted output and real output

we use **cost funtion** to measure the error
![描述](./img/wuenda06.png)

the cost function, also called "the square error function", uses the **least squares method**

m is for averaging, making the function independent of the sample size;<br>
2 is to ensure that the gradient grad after differentiation has no extra coefficients, which cancel out of the square 2

to better visualize the function, let's simplify it first:
![描述](./img/wuenda07.png)
![描述](./img/wuenda08.png)


