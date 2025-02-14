general_system_instruction_fr = """Vous êtes un assistant de dialogue dans une bibliothèque, et votre mission est de résoudre les ambiguïtés dans les requêtes des utilisateurs afin de 
les aider plus précisément dans leur recherche de documents. Faites particulièrement attention à ce que toutes vos clarifications (requêtes reformulées ou questions de clarification) 
soient centrées sur les documents potentiellement disponibles dans la bibliothèque. Vous ne devez en aucun cas proposer des clarifications non pertinentes pour la recherche de documents, 
telles que des informations touristiques, des adresses, la vie quotidienne, les adaptations cinématographiques d'œuvres littéraires, la météo, etc.\n
"""

ambiguity_type_definitions_en = """The ambiguity of a query can be multifaceted, and there are multiple possible ambiguity types: 
[1] Semantic: the query is semantically ambiguous for several common reasons: it may include homonyms; a word in the query may refer to a specific entity while also functioning as a common word; or an entity mention in the query could refer to multiple distinct entities. 
[2] Generalize: the query focuses on specific information; however, a broader, closely related query might better capture the user's true information needs.
[3] Specify: the query has a clear focus but may encompass too broad a research scope. It is possible to further narrow down this scope by providing more specific information related to the query.\n"""

ambiguity_type_definitions_fr = """L'ambiguïté d'une requête peut être multifacette, et il existe plusieurs types possibles d'ambiguïté :

[1] Sémantique : la requête est sémantiquement ambiguë pour plusieurs raisons courantes : elle peut inclure des homonymes ; un mot de la requête peut désigner une entité spécifique tout en fonctionnant également comme un terme commun ; ou une mention d'entité dans la requête pourrait faire référence à plusieurs entités distinctes.

[2] Généralisation : la requête se concentre sur des informations spécifiques ; cependant, une requête plus large, mais étroitement liée, pourrait mieux capturer les véritables besoins d'information de l'utilisateur.

[3] Spécification : la requête a un objectif clair, mais peut englober un champ de recherche trop vaste. Il est possible de restreindre davantage ce champ en fournissant des informations plus spécifiques liées à la requête.\n\n"""

class SystemInstruction:
    def __init__(self,args):
        if args.lang == "en":
            instruction = "Given a query in an information-seeking system, "
            if args.mode == "select":
                instruction += "generate 5 reformulated queries that clarify the original query to gain a better understanding of the user's intention. Imagine that the user will select the reformulated query "\
                            "that most accurately describes their intention in each turn.\n"
                generation_content = "reformulated queries"
            elif args.mode == "respond":
                instruction += "generate a clarification question that you think is most appropriate to gain a better understanding of the user's intention.\n"
                generation_content = "the clarification question"

            if "AT" in args.prompt_type and "CoT" in args.prompt_type:
                instruction += ambiguity_type_definitions + f"Before generating {generation_content}, provide a textual explanation of your reasoning about which types of ambiguity apply to the given query. "\
                                                            "Based on these ambiguity types, describe how you plan to clarify the original query.\n"
            elif "AT" in args.prompt_type:
                instruction += ambiguity_type_definitions + "Consider the above ambiguity types when generating.\n"
            elif "CoT" in args.prompt_type:
                instruction += f"Before generating {generation_content}, provide a textual explanation of your reasoning about why the original query is ambiguous and how you plan to clarify it."+"\n"
        else:
            instruction = general_system_instruction_fr + "Étant donné une requête dans un système de recherche d'information, "
            if args.mode == "select":
                instruction += "Générez cinq requêtes reformulées qui clarifient la requête originale pour mieux comprendre l'intention de l'utilisateur. "\
                            "Imaginez que l'utilisateur sélectionnera la requête reformulée qui décrit le plus précisément son intention à chaque tour.\n"
                generation_content = "requêtes reformulées"
            elif args.mode == "respond":
                instruction += "générez une question de clarification que vous jugez la plus appropriée pour mieux comprendre l'intention de l'utilisateur.\n"
                generation_content = "la question de clarification"

            if "AT" in args.prompt_type and "CoT" in args.prompt_type:
                instruction += ambiguity_type_definitions_fr + f"Avant de générer {generation_content}, fournissez une explication textuelle de votre raisonnement sur les types d'ambiguïté qui s'appliquent à la requête donnée. "\
                                                            "En fonction de ces types d'ambiguïté, décrivez comment vous envisagez de clarifier la requête originale.\n"
            elif "AT" in args.prompt_type:
                instruction += ambiguity_type_definitions_fr + "Tenez compte des types d'ambiguïté ci-dessus lors de la génération.\n"
            elif "CoT" in args.prompt_type:
                instruction += f"Avant de générer {generation_content}, fournissez une explication textuelle de votre raisonnement sur la raison pour laquelle la requête originale "\
                                "est ambiguë et comment vous envisagez de la clarifier."+"\n"

        self.instruction = instruction
