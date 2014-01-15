Bayesian Data Analysis *with PyMC*
======================

###For those in a hurry

My first attempt to adapt to Python (using PyMC) the R code from "Doing Bayesian Data Analysis",
by John K. Krushcke.

####Models done so far:
- Inferring two binomials proportions and their difference;
- Hierarchical prior for Bernoulli likelihood;
- Metric variable for a single group;
- Simple linear regression;
- Oneway ANOVA.

###Quick References
>1. "Doing Bayesian Data Analysis", by John K. Krushcke   
>[http://doingbayesiandataanalysis.blogspot.com.br/](http://doingbayesiandataanalysis.blogspot.com.br/)
>
>2. "Probabilistic Programming and Bayesian Methods for Hackers", by Cam Davidson-Pilon   
>[https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/)
>
>3. "PyMC", by Christopher Fonnesbeck, Anand Patil and David Huard   
>[http://pymc-devs.github.io/pymc/](http://pymc-devs.github.io/pymc/)


###For those with some time to spare

After many years avoiding any book on statistics, I found myself interested on bayesian methods
when I first heard about them in the NLTK (Natural Language ToolKit for Python) Book.
I read some introductory texts on Bayes' Theorem, but even tough the concepts were somewhat clear, I still
wasn't able to understand how useful it was.

Then I found Krushcke's "Doing Bayesian Data Analysis". It was love at first few-pages-glance. Seriously, 
the book is great, specially for those who know nothing about statistics, like myself. The language is 
accessible, there are tons of examples, the hierarchical models are cleverly illustrated using a unique style... 
What are you still doing here? Go buy the book!

Ahem, where was I? Oh, yeah. Krushcke's book. Have I mentioned that it uses R and JAGS (or BUGS) to implement 
the concepts and models? It's really a great deal: you learn bayesian statistics, general probability AND R. 
But I had problems with R syntax. It's so clumsy, and weird and... Well, I don't like it. Great packages,
but I really prefer Python.

Being a Python fan, I had to find a way to adapt the original code. It was made possible after I discovered
PyMC in Cam Davidson-Pilon "Probabilistic Programming and Bayesian Methods for Hackers". Also a great book, with a 
original media: IPython Notebooks - you read the book, you read (and edit!) the code and run it all in your 
favorite browser. Did I mention it is free?

So, as a exercise in bayesian statistics and in Python programming, I adapted some models used by Krushcke 
in his book to Python, using PyMC (and Numpy and Matplotlib). This is just a first attempt, and I have only 
"translated" Krushcke's model to Python - and in a very non-pythonic way. There's plenty of room for improvement, 
but the code shouldn't break if the data is entered correctly. The code is also heavily commented - after all, 
the point here is to exercise the concepts and programming skills.

Feel free to fork and modify and pull request - all suggestions are welcome!
