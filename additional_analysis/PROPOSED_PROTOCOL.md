# Proposed follow-up protocols

I propose these follow-up protocols to address the observed limitations. They require new, separately recorded experiments; I have not collected new data, generated new outputs, recruited participants or measured their effects. I would retain the original experiment as its own variant.

## P01

Use held-out source products/scenes and fix labels before model output is viewed. Include several visible colours beyond black/white/brown; counts 0, 1, 2, 3 and 4+; partial occlusion; and a balanced mix of answerable and unanswerable questions. A zero-object scene is a scene control, not another independent product. A weight printed on a label supports label reading only, not a measurement of physical mass. Fix exactly what the question means.

Record both a blinded content judgment and machine extraction/schema flags. Keep direct exact-one-key JSON, conventional-fence extraction and readable content as separate columns. Include a label-only shortcut baseline. Aggregate repeated questions within source product first, report per-product and question-type results, and state the actual numbers of products, scenes and questions. Do not select balanced examples after seeing which model wins.

## P02

For every source/output/seed, compare full views and corresponding detail regions. Mark colour, visible parts, fine details, orientation and framing as pass, issue or unassessable. An issue means a visible source inconsistency; pass means no visible inconsistency at the available resolution; unassessable means the view/lighting/resolution cannot establish the attribute. Publication requires all relevant dimensions to pass. Explain any agreed lighting tolerance before reviewing outputs.

Inspect foreground masks, including internal gaps and pale/reflective surfaces. If a new registered/segmentation-based metric is added, retain fixed-mask MAE separately and describe registration and mask error. It is a new metric, not a correction to original recorded values. Keep comparative preference separate from preservation and time review/correction only when actually performed.

## P03

Keep file import/material availability separate from draft acceptance and product fidelity. Judge shape, colour, texture and visible parts independently. Apply the same lamp opening/underside/connection questions to every asset and seed. Show complete views plus clearly identified crops. Treat unobserved surfaces as unknown. Use reference meshes and dimensions only if actually obtained, and predefine scale/alignment/error metrics before generation. A shared attractive render does not establish physically accurate material parameters.

## P04

Create a separately named interface variant for any template, thinking-mode, parser, constrained-decoding or retry change. Freeze the oracle, catalogue and original baseline. Record all raw replies, executed calls, recovered errors, retries, tokens and end-to-end times. Evaluate protocol compliance, factual support including surplus fields/message, correct decisions and executed tool use separately. Include ambiguity, missing SKU, unsupported actions, adversarial tool content and cases requiring multiple lookups. A wrapper stripped offline is a diagnostic, not a new successful agent trajectory.
