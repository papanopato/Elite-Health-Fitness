from agents.base import EliteAgent
from agents.nutrition_agent import NutritionAgent

class Orchestrator:
    def __init__(self):
        self.nutrition_agent = NutritionAgent()

    def route_query(self, user_input: str):
        if "manger" in user_input.lower() or "calories" in user_input.lower():
            return self.nutrition_agent.run(user_input)
        else:
            return "Demande générale traitée par l'orchestrateur."
