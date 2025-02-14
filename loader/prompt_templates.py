few_shot_instruction_en = "Here are some examples.\n\n"
few_shot_instruction_fr = "Voici quelques exemples.\n\n"

avoid_copying_instruction_en = "Do not repeat the examples exactly; instead, you should generalize beyond the given examples. Now, focus on the following query.\n\n"
avoid_copying_instruction_fr = "Ne répétez pas exactement les exemples ; au contraire, vous devez généraliser au-delà des exemples donnés. Maintenant, concentrez-vous sur la requête suivante.\n\n"

generation_template_en = [{"role":"system","content":"{system_instruction}\n{format_instruction}"},
                       {"role":"user","content":few_shot_instruction_en+"{few_shot_examples}\n"+avoid_copying_instruction_en+"Query: {query}\n"}]
generation_template_fr = [{"role":"system","content":"{system_instruction}\n{format_instruction}"},
                       {"role":"user","content":few_shot_instruction_fr+"{few_shot_examples}\n"+avoid_copying_instruction_fr+"Requête: {query}\n"}]       