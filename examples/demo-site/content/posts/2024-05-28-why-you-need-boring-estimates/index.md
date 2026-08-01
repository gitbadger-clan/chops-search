+++
title = "Keep your estimates boring"
date = 2024-05-28
time_to_read = 5
[taxonomies]
  tags = ["estimating software projects"]
+++

## Why?

Most everyone I know has got shiny object syndrome. We all want to work on the latest and greatest. I myself am part of that crowd very much. Whenever I start on a project I will never pin the versions for the libraries. That adds anywhere between 0% and 50% on top of the project timeline. The most notable example is Javascript with its myriad of libraries. You would think everyone is familiar with this meme by now :sweat_smile:.

![Zero days without new framework](js-fw-meme.png)

In the first article in the series we talked about getting _anchors_ right and trying to stay away from the uniqueness bias when choosing a good anchor. In _How Big Things Get Done_ the authors also bring in a concept that was new to me: **the reference class**.
The phrase was originally coined in the 1970s by the psychologist [Daniel Kahneman and his colleague Amos Tversky](https://www.newyorker.com/books/page-turner/the-two-friends-who-changed-how-we-think-about-how-we-think) and is regularly used in the context of _reference class forecasting_.
Daniel and Amos refer to two types of views when estimating a project:

- _inside view_ which is the view while working on the project with your personal biases
- _outside view_ which is the view from the outside, looking at the project as a whole and comparing it to similar projects

Enough with the theory, in practice, what we actually want to achieve is to have our estimates work and be as accurate as possible.

## How?

Now that we have the lingo down, let's get into the nitty-gritty. We want to figure out how to calculate the estimates, you got that right **calculate**. The calculations are based on being able to cut down your project from being a special and unique snowflake to a project that is similar to others. It's a combination of statistical and historical analysis of other projects as similar as possible to yours.

---

_Going on a tangent here, wouldn't it be cool if we could have a database of software project estimates with numerical data, situational requirements and conditions, and perhaps how long the project took in the end?_ :thinking:

---

You want to reduce your project to something as generic as possible then look for data about other projects like it. As a software developer you may be tempted to think that it is special and unique, but finding the commonalities will help you get a better estimate.
We could make use of both _inside view_ and _outside view_:

- _outside view_: see how long similar projects took(take the median) - I am referring to the reduced version where you cut out any product differentiators
- _inside view_: see how long the differentiators will take (_this you can break down further as well into common tasks and unique tasks_)

... do you see where I am going with this? It's all turtles all the way down :turtle:. Now you can already see how things can be broken down further into smaller pieces and how similarities can make estimating easier.

The numbers show a 30% increase in accuracy when using _reference class forecasting_, that is the _outside view_, with 50% not being uncommon.

The aspect that is different from plain anchor-based estimates is that you choose an anchor that is based on the _reference class_ which makes it closer to the objective reality.

## Into the future with AI

Last couple of years I have been working in the field of AI and I have been an avid reader of various papers and consumed a decent amount of tutorials and courses. Deep learning is an amazingly powerful tool that is able to draw conclusions based on the importance of a particular feature of the project and classify it.

If we had the data about projects we could train a multi-class classifier to predict the time it would take to complete a project(S/M/L/XL). This could be a great Trello plugin for example.

Linear regression can be another simpler approach to do a numeric estimate of the project timeline. Now, this feels like we are taking all the joy out of the agile SDLC, but remember this is only supposed to be used as a data focused approach, from the _outside view_ i.e. looking objectively at the data, so no hard feelings to be had :wink:.

Thinking a bit further we could have an LLM + RAG system that looks at the database of projects we have broken down, does a similarity search and gives us some kind of standard estimates.

The data would probably be a huge challenge for this one. You would have to get data from various sources and have it clean, usable and can be used to train the models.

## Conclusion

- **Stay boring**: don't get caught up in the thinking your project is special and unique
- **Use the reference class**: look at similar projects and see how long they took
- **Use both views**: _inside view_ and _outside view_ to get a better estimate
- **Use data**: if you have it, use it to your advantage
- **Ask for help**: if you are not sure, ask someone who has done it before, outside perspective can add a layer of objectivity
