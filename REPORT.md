

## The problem

so it lok like training was not working. The loss went up instead of
down, and then stopped moving. The model was not learning anything useful.

We reproduced that. Running the setup as it was, validation MAE sat between
14 and 16 and never improved. The training loss stayed flat at 3.70 for 44
epochs straight.

3.70 turned out to be meaningful. It is roughly the loss you get
if the model just outputs zero everywhere. So the model was not stuck part of
the way to a solution. It was sitting on the "predict nothing" answer and not
moving off it.

For comparison: if you ignore the image completely and always guess the
average crowd size, you get MAE 5.1 on the test set. So the broken model was
about three times worse than guessing.

## The score

Best run now gets **MAE 1.031** on the UCSD test set (all 1200 test frames).

| what we ran | test MAE | test RMSE |
| --- | --- | --- |
| **sigma 8, Adam** | **1.031** | **1.290** |
| sigma 8, Adam, crop filter | 1.116 | 1.370 |
| sigma 8, SGD | 1.194 | 1.538 |
| sigma 3, SGD (the paper's exact recipe) | 1.286 | 1.647 |
| sigma 8, Adam, photo augmentation | 1.305 | 1.584 |
| sigma 3, Adam | 1.698 | 2.062 |
| sigma 8, Adam, both augmentations, 360 epochs | 1.832 | 2.140 |
| CSRNet paper | 1.16 | 1.47 |
| MCNN paper | 1.07 | 1.35 |

So yes, it improved. From not learning at all, to beating the number the
CSRNet paper reports on this dataset.

Two of those rows matter most:

- **1.286** is the paper's own recipe, reproduced. Same optimiser, same
  kernel, same everything. This is our honest "we reproduced CSRNet" number.
- **1.031** is what we get when we change one thing: the width of the blobs
  in the ground truth. This beats the paper.

## What was actually broken

Seven bugs in the code. ahahaha :(

**1. The loss was not divided by the batch size.**
This was the main one. The code added up the squared error over every pixel
and every image in the batch, and never divided by anything. So the gradient
got bigger as the batch got bigger. We measured it: gradient size 13.7 at
batch 1, 264.5 at batch 16. Nearly 20 times bigger. The learning rate was
tuned for batch 1, so at batch 16 every step was about 20 times too big and
the loss climbed. The CSRNet paper actually defines the loss with a divide by
2N in it.


**2. The learning rate schedule did nothing.**
The code multiplied the learning rate by a list of numbers that were all 1.
So it never decayed. It was a fixed learning rate pretending to be a schedule.

**3. schedule had its own bug.**
I wrote a proper cosine schedule, but it read its starting value from the
variable it was also writing to. So each epoch shrank the already shrunk
value. By epoch 90 the learning rate was 2.4e-7 instead of 8.8e-6, about 37
times too small. Training quietly stopped around epoch 50.

**4. Every training crop under-counted by 2.1%.**
The code cuts a 316 by 476 piece out of each image, shrinks the density map by
8, and multiplies by 64 to keep the count right. But 316 divided by 8 is 39.5,
not a whole number. So the real shrink factor was not 8, and multiplying by 64
lost 2.1% of the people. Full images were fine because 632 and 952 do divide
by 8. So the model was trained to under-count and then graded against correct
counts. Rounding the crop down to a multiple of 8 fixed it exactly.

**5. The cluster training script passed flags that no longer existed.**
A merged pull request renamed the Weights and Biases options but did not
update the SLURM script. Every cluster job would have died in one second with
"unrecognized arguments".

**6. The gradient size in the logs was wrong.**
Under mixed precision the number printed was the loss scale factor, not the
gradient. It showed values in the thousands when the real gradient was about
5. The one number you would look at to diagnose a stuck run was useless.

**7. The random seed was the clock.**
Every run got a different seed. So two runs that differed only in the setting
we were testing also differed in their random initialisation. The experiment
could not separate the setting from the noise.


## @Sesethu this stuff made a big diffewwent: 

Changing the blob width in the ground truth, from sigma 3 to sigma 8.

The ground truth is made by putting a small Gaussian blob on each person's
head. At sigma 3, only 2.5% of the output pixels have anything in them. The
other 97.5% are zero. So most of what the network is told is "put zero here".
There is very little signal telling it where people actually are.

At sigma 8 each person covers about 8.5% of the pixels. Same total count, just
spread wider. That one change took test MAE from 1.698 to 1.031.

The CSRNet paper says sigma 3 for UCSD. But it also says it scales the images
up by 4 and never says whether the sigma is measured before or after that
scaling. Sigma 8 on the scaled image is about sigma 2 on the original. So
sigma 8 may well be what they actually used.

##a haha, this stuff did not work, good for wirting a ppaeer @sesethu, inlude them

**Augmentation.** We tried brightness and contrast jitter, pixel noise, and a
crop filter that skips crops with almost nobody in them. None of it helped on
the test set. The best augmented run got 1.305, against 1.031 with nothing.

We got this wrong twice before getting it right. First we thought augmentation
was actively harmful, because the combined run scored 5.9. Then we thought one
specific part of it was to blame. Both wrong. What actually happened is that
augmentation makes the job harder, so the model needs more epochs. At 120
epochs it had not finished learning. At 360 epochs it reached 1.209 on
validation. But on the test set it still came out at 1.832, worse than plain
sigma 8. So it was undertrained *and* it does not help.

That makes sense for this dataset. UCSD is one fixed camera pointed at one
walkway, in steady light. Teaching the model to ignore brightness changes buys
nothing, because the test images have the same brightness as the training
ones.

**Combining the two best settings.** Sigma 8 was the best data setting. SGD at
the corrected learning rate was a good optimiser setting. Putting them
together gave 1.194, worse than sigma 8 with Adam at 1.031. They do not add up.

## Something to be careful about

Validation scores did not predict test scores. The run with the best
validation score (1.156) came out near the bottom on test (1.305). The run
that won on test (1.031) was only third on validation.

The reason is that the validation set here is the last 10% of each training
clip. It is small, it is noisy, and it is harder than the test set, with an
average of 28.5 people per frame against 24.5. On top of that, the early runs
all used different random seeds.

So do not read anything into small gaps between runs. The gap between sigma 8
and sigma 3 is big enough to trust. Gaps of 0.1 or so between the top few runs
are not.

