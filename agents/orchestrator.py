from agents.base import EliteAgent

class Orchestrator:
    def __init__(self):
        # Initialisation des agents (seront configurés plus tard)
        self.nutrition_agent = None 
        self.fitness_agent = None

    def route_query(self, user_input: str):
        """
        Logique de routage : décide quel agent utiliser selon la question.
        """
        if "manger" in user_input.lower() or "calories" in user_input.lower():
            return "Redirection vers l'agent Nutrition..."
        elif "sport" in user_input.lower() or "exercice" in user_input.lower():
            return "Redirection vers l'agent Fitness..."
        else:
            return "Demande générale traitée par l'orchestrateur."