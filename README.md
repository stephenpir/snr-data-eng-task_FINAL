My home dev setup uses Python 3.10.9 and a requirements.txt is provided at the project root
My solution is in the "Stephen_Pir_answers" directory with Parts and exericise number clearly numbered in file titles and code comments.
The code is heavily commented for readability and to give indication of my thoughts.
I have also provided a requests.http with the data from the produced training set


1. What part of the exercise did you find most challenging, and why?
Part 2 was more challenging. I've not used FastAPI in anger before so had to brush up on that before even starting.
Unfortunately, I also encountered some incompatibility when trying to use FastAPI and the provided model.joblib.
Initially I thought an old version of some module was used in the joblib file as other sample code provided contained deprecated methods. 
I did try to ascertain version info from the lib file provided but there wasn’t any metadata of this nature.
However, enhancing error handling, further investigation and some help from AMP, showed this was a development environments issue, as I’m on an ARM Mac (M-series chip). 
Uninstalling and reinstalling some modules (numpy and scikit-learn) eventually resolved the issue.
Annoying, fiddly, and time consuming problem, but I got there in the end! :)

2. What tradeoffs did you make? (e.g., speed vs. accuracy, simplicity vs. completeness)
With the size of the sample data, speed wasn't really too much of a consideration if we are talking about runtime performance.
In terms of completeness, the code is pretty robust due to using Gemini for the framework. 
Implementing data checks but not actual clensing as per requirement feels incomplete though.

3. Assume this needs to run in production with these constraints:
– Cloud provider: Azure
– Budget: £500/month
– Latency requirement: <100ms per prediction
– Expected traffic: 1000 predictions/hour initially
What would you improve or change first?

Should have it as a containerised microservice to minimise cost footprint.
The app/ model should be remain loaded after startup, rather than loaded for each prediction and stored with fast access in mind.
Validation of the messages should be done up front rather than when submitted to the API to minimise traffic/ latency.
Maybe region specifc instances to minimise network latency

4. How would you deploy the FastAPI service and make the model artifact available?
Use Azure Blob storage

5. If transaction volume jumped from thousands to millions per day, how would you rethink Part 1?
Switch from batch to streaming ingestion with kafka topics and store the relevant period of transactions/ aggregates per customer so rolling calculation is less faster.
Scale with multiple instances of the service possibly.

6. What metrics would you track in production and why? What could go wrong with this model in production?
Metrics: 
Application Health
Latency and throughput	
Accurancy of predictions

What could go wrong:
Oversubscription causing latency
Application failures
Schema changes causing failure in calls to the API
Model may be inaccurate so should be checked

7. If you used AI tools such as ChatGPT, Claude, Copilot, or any other tools:
– Where and how did you use them? (e.g., boilerplate code, debugging, syntax help)
– How did they help or hinder your process?

I used gemini to do initial drafts of code as it has a good free usage quota, though I really like AMP I save it for more complex deep dives (like the environments issue I had in Part 2), due to limited usage quota. 
After initial draft I read through and amended the code to ensure it performed the task as desired. 
Areas for improvement included:
-	I probably could’ve used stronger prompting for better initial results in terms of what was desired/ needed generally but overall structure generated was pretty good.
-	Over engineering the solution, giving far more code than needed. Although robust it hampered readability, so changes had to be made there.
-	Code refactoring was needed to prevent overkill e.g. gemini separated out the aggregations rather than doing them all in one pass with lamda functions
-	Gemini seemed to be more interested in the counts than the amounts when trying to make inferences about the data. Clearly in a system about debt and likelihood of default it’s more interesting to know how much, and on what, customers spend than how many transactions they made.

