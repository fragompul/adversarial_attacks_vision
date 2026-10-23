# Attacks

This folder implements every adversarial attack used elsewhere in the repo, split by threat model and by what kind of claim the attack actually makes. `whitebox/` assumes full access to the model's gradients; `blackbox/` assumes only query access to its outputs, a far more realistic assumption for an attacker facing a deployed API; `physical/` drops the digital-perturbation assumption entirely and asks whether an attack can survive being printed and photographed. Three further subfolders each break a different assumption the first three still share: `universal/` asks whether one perturbation can work across many images instead of being fit per image; `semantic/` asks whether an attack needs to touch pixel color at all, rather than where content sits in the frame; `poisoning/` moves the attack out of inference time entirely, into the training data itself. Reading `whitebox/` through `physical/` first builds up the inference-time threat model from most to least permissive access; the last three each relax a different assumption those three still share.

## whitebox/

`01_FGSM` implements the Fast Gradient Sign Method (Goodfellow et al., 2014), a single gradient step in the direction that most increases the model's loss. It is the cheapest attack in the repo and the one every other whitebox method here is measured against.

`02_PGD` turns FGSM into an iterative attack: several small gradient steps, each one projected back onto the epsilon ball around the original image, which makes it considerably stronger at the same perturbation budget.

`03_DeepFool` takes a different approach, linearizing the decision boundary around the input and stepping towards the closest one; the perturbation it finds is close to the minimal one needed to cause a misclassification, which makes it useful as a lower bound on how little distortion an attack actually needs.

`04_C&W` (Carlini & Wagner) formulates the attack as an optimization problem in a tanh-transformed space, directly minimizing L2 distortion subject to a misclassification constraint. It is the slowest attack in the repo, and generally considered the strongest L2 attack against undefended models.

`05_TargetedIFGSM` extends the iterative FGSM idea to targeted attacks, pushing the prediction towards a specific chosen class rather than just away from the correct one, stopping once a target confidence threshold is reached.

## blackbox/

These three attacks never touch a gradient; they only see the model's output for each query, which is the realistic setting for an attacker probing a deployed API rather than a model they own.

`01_SquareAttack` is score-based: it needs the full probability vector and searches greedily over random square-shaped perturbations, keeping any change that lowers the true class's margin. It was run at real scale across the three ImageNet models, two sample images, and query budgets from 50 to 1000.

`02_NES` estimates the gradient itself from queries, using natural evolution strategies, and is considerably more expensive per query than Square Attack since each one requires sampling a population of noise directions. It timed out at the original budget on this machine's CPU and was re-run at a reduced scale, one image with budgets up to 500, documented in the notebook itself.

`03_BoundaryAttack` is decision-based rather than score-based: it only needs the predicted label, not a confidence score, the hardest and most realistic threat model of the three. It starts from a large perturbation known to be adversarial and walks it back towards the original image along the decision boundary. Also reduced in budget after an initial timeout, for the same CPU-cost reasons as NES.

`04_OnePixelAttack` bounds the perturbation by how many pixels can change ($L_0$) instead of how much any one pixel can move, found with differential evolution rather than a gradient. See the folder's results directly for how this plays out at ImageNet resolution, a very different regime from the low-resolution images the original paper demonstrates it on.

## physical/

`01_AdversarialPatch` moves away from imperceptible, whole-image perturbations towards a small, visible patch trained with Expectation over Transformation (EOT) so it survives random scale, rotation and position, the same threat model Eykholt et al. (2018) used against real stop signs. Trained on TrafficNet/GTSRB, targeted from the Stop class to "Ahead only", it reaches a 47.5% attack success rate over placements never seen during the patch's own training.

## universal/, semantic/, poisoning/

Each of these breaks one assumption the six notebooks above still share. `universal/` asks whether a single, fixed perturbation can fool a model across many different images, not just the one it was optimized against. `semantic/` asks whether an attack needs to change pixel color at all, warping the image instead. `poisoning/` moves the attack out of inference entirely, into the training data itself, a completely different stage of the model's lifecycle. See each folder's own README for what was actually found; none of the three fit naturally into the whitebox/blackbox/physical framework above, which is exactly why they get their own.

## A note on what is not here

NES and Boundary Attack are excluded from the wider attack-comparison benchmarks elsewhere in the repo (for instance in `defenses/04_adversarial_training.ipynb`) purely for cost reasons: at hundreds of images across multiple models, their per-image query cost would dominate the runtime for a comparatively small addition to the result. They remain fully implemented and validated here, just evaluated at a smaller scale that keeps them tractable on a CPU.
