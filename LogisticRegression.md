# What Logistic Regression actually is

Forget the scary math name for a second. It's not really about trees or questions — it's about weighing numbers.

Imagine you're guessing fraud by hand, and you assign a "suspicion score" to each fact about a claim:

Big gap between claimed and approved amount → +40 suspicion points
Claim status is "Rejected" → +30 points
Patient has a chronic condition → −10 points (less suspicious, makes sense medically)
Claim submitted same day → +15 points

You add all these up, and if the total score is high enough, you call it fraud. That's basically what Logistic Regression does — except instead of a human guessing the point values, the algorithm learns the best point values (called "weights") by looking at thousands of past claims and adjusting until its scores line up with which ones were actually fraud.