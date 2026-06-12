from agents.base import EliteAgent

class NutritionAgent(EliteAgent):
    def __init__(self):
        super().__init__(
            name="Nutrition Elite",
            instructions=[
                "Tu es un expert en nutrition sportive.",
                "Tu calcules les besoins caloriques et suggères des repas équilibrés.",
                "Tu réponds toujours de manière concise et motivante."
            ]
        )
