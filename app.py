"""
AI Health & Fitness Planner — Version Optimisée
Améliorations : appels parallèles, structuration JSON, champs dynamiques, validation
"""

import streamlit as st
import json
import concurrent.futures
from typing import Dict, Any, Optional
from dataclasses import dataclass
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.models.google import Gemini


# ───────────────────────────────────────────────
# CONFIGURATION & MODÈLES DE DONNÉES
# ───────────────────────────────────────────────

@dataclass
class UserProfile:
    """Profil utilisateur typé et validé"""
    age: int
    weight: float
    height: float
    sex: str
    activity_level: str
    dietary_preferences: str
    fitness_goals: str

    @property
    def bmi(self) -> float:
        return round(self.weight / ((self.height / 100) ** 2), 1)

    @property
    def bmr(self) -> float:
        """Mifflin-St Jeor equation"""
        if self.sex == "Male":
            return round(10 * self.weight + 6.25 * self.height - 5 * self.age + 5, 0)
        elif self.sex == "Female":
            return round(10 * self.weight + 6.25 * self.height - 5 * self.age - 161, 0)
        else:
            return round(10 * self.weight + 6.25 * self.height - 5 * self.age - 78, 0)

    @property
    def tdee(self) -> float:
        """Total Daily Energy Expenditure"""
        multipliers = {
            "Sedentary": 1.2,
            "Lightly Active": 1.375,
            "Moderately Active": 1.55,
            "Very Active": 1.725,
            "Extremely Active": 1.9
        }
        return round(self.bmr * multipliers.get(self.activity_level, 1.2), 0)

    def to_prompt(self) -> str:
        return f"""User Profile:
- Age: {self.age} years
- Weight: {self.weight} kg
- Height: {self.height} cm
- Sex: {self.sex}
- BMI: {self.bmi}
- BMR: {self.bmr} kcal/day
- TDEE: {self.tdee} kcal/day
- Activity Level: {self.activity_level}
- Dietary Preferences: {self.dietary_preferences}
- Fitness Goals: {self.fitness_goals}"""


# ───────────────────────────────────────────────
# PROMPTS STRUCTURÉS (JSON Schema)
# ───────────────────────────────────────────────

DIETARY_PROMPT = """You are an expert nutritionist. Based on the user profile provided, generate a personalized dietary plan.

CRITICAL: Respond ONLY with a valid JSON object matching this exact structure:
{
    "why_this_plan_works": "Detailed explanation tailored to the user's specific profile, goals, and dietary preferences. Explain the science behind the macronutrient distribution.",
    "daily_calories_target": <number>,
    "macros": {
        "protein_g": <number>,
        "carbs_g": <number>,
        "fats_g": <number>
    },
    "meal_plan": {
        "breakfast": {"items": ["item1", "item2"], "calories": <number>, "notes": "preparation tip"},
        "lunch": {"items": ["item1", "item2"], "calories": <number>, "notes": "preparation tip"},
        "dinner": {"items": ["item1", "item2"], "calories": <number>, "notes": "preparation tip"},
        "snacks": {"items": ["item1", "item2"], "calories": <number>, "notes": "timing suggestion"}
    },
    "hydration": {
        "daily_target_liters": <number>,
        "schedule": "Hour-by-hour hydration schedule",
        "electrolyte_strategy": "Specific electrolyte recommendations based on activity level"
    },
    "important_considerations": [
        "Consideration 1 specific to user's dietary preference",
        "Consideration 2 specific to user's goal",
        "Consideration 3 about micronutrients",
        "Consideration 4 about meal timing"
    ],
    "supplements_if_needed": ["supplement1", "supplement2"]
}

Rules:
- Adapt portions to the user's TDEE and goals (deficit for weight loss, surplus for muscle gain)
- Respect ALL dietary restrictions strictly
- Include specific gram amounts, not vague portions
- Consider the user's activity level for carb timing"""

FITNESS_PROMPT = """You are an expert personal trainer. Based on the user profile provided, generate a personalized fitness plan.

CRITICAL: Respond ONLY with a valid JSON object matching this exact structure:
{
    "goals": "Specific, measurable goals tailored to the user's profile (e.g., 'Lose 5kg in 12 weeks through combined cardio and resistance training')",
    "training_split": "Weekly schedule (e.g., 'Upper/Lower/Rest/Upper/Lower/Active Recovery/Rest')",
    "weekly_schedule": [
        {"day": "Monday", "focus": "muscle group", "duration_min": <number>, "intensity": "Low/Medium/High"},
        {"day": "Tuesday", "focus": "muscle group", "duration_min": <number>, "intensity": "Low/Medium/High"}
    ],
    "warm_up": {
        "duration_min": <number>,
        "exercises": [
            {"name": "exercise name", "sets": <number>, "reps": "duration or count", "purpose": "why this exercise"}
        ]
    },
    "main_workout": {
        "exercises": [
            {"name": "exercise name", "sets": <number>, "reps": "count", "rest_sec": <number>, "target_muscles": ["muscle1", "muscle2"], "progression": "how to advance"}
        ]
    },
    "cool_down": {
        "duration_min": <number>,
        "exercises": ["stretch1", "stretch2"],
        "breathing_technique": "specific technique"
    },
    "cardio_recommendations": {
        "weekly_sessions": <number>,
        "type": "LISS/HIIT/MISS based on goal",
        "duration_min": <number>,
        "heart_rate_zone": "target zone"
    },
    "tips": [
        "Tip 1 specific to user's experience level",
        "Tip 2 about progressive overload",
        "Tip 3 about recovery matching activity level",
        "Tip 4 about form and injury prevention"
    ],
    "progress_tracking": {
        "metrics": ["metric1", "metric2"],
        "frequency": "how often to assess",
        "benchmarks": ["benchmark1", "benchmark2"]
    }
}

Rules:
- Scale difficulty to the user's age, BMI, and activity level
- For sedentary users: start conservative, focus on movement habits
- For very active users: include periodization
- Include specific sets, reps, and rest periods
- Consider joint health for users with high BMI"""


# ───────────────────────────────────────────────
# FONCTIONS D'AGENTS (PARALLÈLES)
# ───────────────────────────────────────────────

def create_dietary_agent(model) -> Agent:
    return Agent(
        name="Dietary Expert",
        role="Expert nutritionist creating data-driven meal plans",
        model=model,
        instructions=[DIETARY_PROMPT],
        markdown=False,  # Force raw text for JSON parsing
        debug_mode=False
    )

def create_fitness_agent(model) -> Agent:
    return Agent(
        name="Fitness Expert",
        role="Expert personal trainer creating structured workout programs",
        model=model,
        instructions=[FITNESS_PROMPT],
        markdown=False,
        debug_mode=False
    )

def run_agent_safe(agent: Agent, profile: UserProfile) -> Dict[str, Any]:
    """Exécute un agent avec gestion d'erreurs et parsing JSON"""
    try:
        response: RunOutput = agent.run(profile.to_prompt())
        content = response.content if hasattr(response, 'content') else str(response)

        # Nettoyage du markdown JSON
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parsing failed: {str(e)}", "raw_response": content}
    except Exception as e:
        return {"error": str(e)}

def generate_plans_parallel(model, profile: UserProfile) -> tuple[Dict, Dict]:
    """Exécute les deux agents en PARALLÈLE via ThreadPool"""
    dietary_agent = create_dietary_agent(model)
    fitness_agent = create_fitness_agent(model)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_diet = executor.submit(run_agent_safe, dietary_agent, profile)
        future_fitness = executor.submit(run_agent_safe, fitness_agent, profile)

        dietary_plan = future_diet.result(timeout=60)
        fitness_plan = future_fitness.result(timeout=60)

    return dietary_plan, fitness_plan


# ───────────────────────────────────────────────
# AFFICHAGE STREAMLIT (UI AMÉLIORÉE)
# ───────────────────────────────────────────────

def display_metrics(profile: UserProfile):
    """Affiche les métriques calculées du profil"""
    cols = st.columns(4)
    metrics = [
        ("BMI", profile.bmi, "kg/m²"),
        ("BMR", f"{profile.bmr:.0f}", "kcal/day"),
        ("TDEE", f"{profile.tdee:.0f}", "kcal/day"),
        ("Goal", profile.fitness_goals, "")
    ]
    for col, (label, value, unit) in zip(cols, metrics):
        with col:
            st.metric(label, f"{value} {unit}".strip())

def display_dietary_plan(plan: Dict):
    if "error" in plan:
        st.error(f"❌ Dietary plan error: {plan['error']}")
        if "raw_response" in plan:
            with st.expander("Raw response"):
                st.code(plan["raw_response"])
        return

    with st.expander("📋 Your Personalized Dietary Plan", expanded=True):
        # Explication personnalisée
        st.markdown("### 🎯 Why This Plan Works")
        st.info(plan.get("why_this_plan_works", "No explanation available"))

        # Métriques nutritionnelles
        cols = st.columns(3)
        macros = plan.get("macros", {})
        cols[0].metric("Protein", f"{macros.get('protein_g', 0)}g")
        cols[1].metric("Carbs", f"{macros.get('carbs_g', 0)}g")
        cols[2].metric("Fats", f"{macros.get('fats_g', 0)}g")
        st.caption(f"Daily target: **{plan.get('daily_calories_target', 'N/A')} kcal**")

        # Plan de repas structuré
        st.markdown("### 🍽️ Meal Plan")
        meal_plan = plan.get("meal_plan", {})
        for meal_name, meal_data in meal_plan.items():
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{meal_name.capitalize()}** — {meal_data.get('calories', '?')} kcal")
                    for item in meal_data.get("items", []):
                        st.write(f"• {item}")
                    if meal_data.get("notes"):
                        st.caption(f"💡 {meal_data['notes']}")

        # Hydratation
        hydration = plan.get("hydration", {})
        with st.container(border=True):
            st.markdown("### 💧 Hydration Strategy")
            st.write(f"**Target:** {hydration.get('daily_target_liters', '?')}L/day")
            st.write(f"**Schedule:** {hydration.get('schedule', 'N/A')}")
            st.write(f"**Electrolytes:** {hydration.get('electrolyte_strategy', 'N/A')}")

        # Considérations importantes
        st.markdown("### ⚠️ Important Considerations")
        for consideration in plan.get("important_considerations", []):
            st.warning(consideration)

        # Suppléments
        supplements = plan.get("supplements_if_needed", [])
        if supplements:
            st.markdown("### 💊 Recommended Supplements")
            st.write(", ".join(supplements))

def display_fitness_plan(plan: Dict):
    if "error" in plan:
        st.error(f"❌ Fitness plan error: {plan['error']}")
        if "raw_response" in plan:
            with st.expander("Raw response"):
                st.code(plan["raw_response"])
        return

    with st.expander("💪 Your Personalized Fitness Plan", expanded=True):
        # Objectifs personnalisés
        st.markdown("### 🎯 Goals")
        st.success(plan.get("goals", "Goals not specified"))

        # Split hebdomadaire
        st.markdown(f"### 📅 Training Split: *{plan.get('training_split', 'N/A')}*")
        schedule = plan.get("weekly_schedule", [])
        if schedule:
            days_data = {s["day"]: s for s in schedule}
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            cols = st.columns(7)
            for col, day in zip(cols, days):
                with col:
                    day_info = days_data.get(day, {})
                    emoji = "💪" if day_info else "🛌"
                    st.markdown(f"**{emoji} {day[:3]}**")
                    if day_info:
                        st.caption(f"{day_info.get('focus', 'Rest')}")
                        st.caption(f"{day_info.get('duration_min', '?')} min")

        # Warm-up
        warm_up = plan.get("warm_up", {})
        with st.container(border=True):
            st.markdown(f"### 🔥 Warm-up ({warm_up.get('duration_min', '?')} min)")
            for ex in warm_up.get("exercises", []):
                st.write(f"• **{ex.get('name', '?')}** — {ex.get('sets', '?')} sets × {ex.get('reps', '?')} ({ex.get('purpose', '')})")

        # Main workout
        main = plan.get("main_workout", {})
        with st.container(border=True):
            st.markdown("### 🏋️‍♂️ Main Workout")
            for ex in main.get("exercises", []):
                with st.container():
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        muscles = ", ".join(ex.get("target_muscles", []))
                        st.write(f"**{ex.get('name', '?')}** — {muscles}")
                        st.caption(f"Progression: {ex.get('progression', 'N/A')}")
                    with c2:
                        st.write(f"{ex.get('sets', '?')} × {ex.get('reps', '?')}")
                        st.write(f"Rest: {ex.get('rest_sec', '?')}s")

        # Cool-down
        cool = plan.get("cool_down", {})
        with st.container(border=True):
            st.markdown(f"### 🧘 Cool-down ({cool.get('duration_min', '?')} min)")
            st.write(", ".join(cool.get("exercises", [])))
            st.caption(f"Breathing: {cool.get('breathing_technique', 'N/A')}")

        # Cardio
        cardio = plan.get("cardio_recommendations", {})
        with st.container(border=True):
            st.markdown("### 🏃 Cardio")
            st.write(f"**{cardio.get('weekly_sessions', '?')}×/week** — {cardio.get('type', 'N/A')}")
            st.write(f"Duration: {cardio.get('duration_min', '?')} min | Zone: {cardio.get('heart_rate_zone', 'N/A')}")

        # Tips personnalisés
        st.markdown("### 💡 Pro Tips")
        for tip in plan.get("tips", []):
            st.info(tip)

        # Suivi de progression
        tracking = plan.get("progress_tracking", {})
        with st.container(border=True):
            st.markdown("### 📊 Progress Tracking")
            st.write(f"**Metrics:** {', '.join(tracking.get('metrics', []))}")
            st.write(f"**Frequency:** {tracking.get('frequency', 'N/A')}")
            st.write(f"**Benchmarks:** {', '.join(tracking.get('benchmarks', []))}")


# ───────────────────────────────────────────────
# Q&A AMÉLIORÉ (RAG-like context)
# ───────────────────────────────────────────────

def create_qa_agent(model, dietary_plan: Dict, fitness_plan: Dict) -> Agent:
    """Crée un agent Q&A avec contexte enrichi des deux plans"""

    # Construction d'un contexte structuré et concis
    diet_summary = json.dumps({
        "calories": dietary_plan.get("daily_calories_target", "N/A"),
        "macros": dietary_plan.get("macros", {}),
        "restrictions": dietary_plan.get("important_considerations", [])[:2],
        "hydration": dietary_plan.get("hydration", {}).get("daily_target_liters", "N/A")
    }, indent=2)

    fitness_summary = json.dumps({
        "goals": fitness_plan.get("goals", "N/A"),
        "split": fitness_plan.get("training_split", "N/A"),
        "cardio": fitness_plan.get("cardio_recommendations", {}),
        "tips": fitness_plan.get("tips", [])[:2]
    }, indent=2)

    context = f"""You are a health and fitness consultant answering questions about a personalized plan.

DIETARY PLAN SUMMARY:
{diet_summary}

FITNESS PLAN SUMMARY:
{fitness_summary}

INSTRUCTIONS:
- Answer based ONLY on the provided plans
- If the question is outside the scope, say so politely
- Be concise but informative
- Use markdown formatting for readability"""

    return Agent(
        name="Health Consultant",
        role="Answers questions about personalized health and fitness plans",
        model=model,
        instructions=[context],
        markdown=True
    )


# ───────────────────────────────────────────────
# APPLICATION PRINCIPALE
# ───────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AI Health & Fitness Planner",
        page_icon="🏋️‍♂️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS personnalisé
    st.markdown("""
    <style>
        .main-header { font-size: 2.5rem; font-weight: 700; color: #1E88E5; }
        .metric-card { background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1E88E5; }
        .stExpander { border: 1px solid #e0e0e0; border-radius: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

    # Initialisation session state
    if 'plans_generated' not in st.session_state:
        st.session_state.plans_generated = False
        st.session_state.dietary_plan = {}
        st.session_state.fitness_plan = {}
        st.session_state.qa_pairs = []
        st.session_state.profile = None

    st.title("🏋️‍♂️ AI Health & Fitness Planner")
    st.markdown("*Personalized plans powered by Gemini 2.5 Flash*")

    # ── SIDEBAR ──
    with st.sidebar:
        st.header("🔑 API Configuration")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get your key at aistudio.google.com/apikey"
        )

        if not gemini_api_key:
            st.warning("⚠️ Enter your Gemini API Key to proceed")
            st.markdown("[Get free API key](https://aistudio.google.com/apikey)")
            return

        try:
            gemini_model = Gemini(id="gemini-2.5-flash-preview-05-20", api_key=gemini_api_key)
            st.success("✅ API Key valid — Gemini 2.5 Flash ready")
        except Exception as e:
            st.error(f"❌ API Error: {e}")
            return

        st.divider()
        st.header("👤 Your Profile")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 10, 100, 30, 1)
            height = st.number_input("Height (cm)", 100.0, 250.0, 175.0, 0.1)
            activity = st.selectbox("Activity", 
                ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"])
            diet = st.selectbox("Diet", 
                ["Vegetarian", "Keto", "Gluten Free", "Low Carb", "Dairy Free", "Omnivore", "Vegan"])

        with col2:
            weight = st.number_input("Weight (kg)", 20.0, 300.0, 70.0, 0.1)
            sex = st.selectbox("Sex", ["Male", "Female", "Other"])
            goal = st.selectbox("Goal", 
                ["Lose Weight", "Gain Muscle", "Endurance", "Stay Fit", "Strength Training", "Body Recomposition"])

        # Bouton de génération
        generate = st.button("🎯 Generate Plan", use_container_width=True, type="primary")

    # ── GÉNÉRATION DES PLANS ──
    if generate:
        profile = UserProfile(age, weight, height, sex, activity, diet, goal)
        st.session_state.profile = profile

        with st.spinner("⏳ Generating your personalized plans in parallel..."):
            try:
                dietary, fitness = generate_plans_parallel(gemini_model, profile)
                st.session_state.dietary_plan = dietary
                st.session_state.fitness_plan = fitness
                st.session_state.plans_generated = True
                st.session_state.qa_pairs = []
                st.rerun()
            except Exception as e:
                st.error(f"❌ Generation failed: {e}")

    # ── AFFICHAGE DES PLANS ──
    if st.session_state.plans_generated and st.session_state.profile:
        profile = st.session_state.profile

        # Métriques du profil
        st.subheader("📊 Your Calculated Metrics")
        display_metrics(profile)
        st.divider()

        # Plans
        display_dietary_plan(st.session_state.dietary_plan)
        display_fitness_plan(st.session_state.fitness_plan)

        # ── Q&A SECTION ──
        st.divider()
        st.header("❓ Questions about your plan?")

        col_q, col_btn = st.columns([4, 1])
        with col_q:
            question = st.text_input("Ask anything about your diet or workout", 
                placeholder="e.g., Can I swap chicken for tofu?")
        with col_btn:
            st.write("")  # spacer
            st.write("")
            ask = st.button("Get Answer", use_container_width=True)

        if ask and question:
            with st.spinner("🤔 Analyzing your question..."):
                try:
                    qa_agent = create_qa_agent(
                        gemini_model,
                        st.session_state.dietary_plan,
                        st.session_state.fitness_plan
                    )
                    response: RunOutput = qa_agent.run(question)
                    answer = response.content if hasattr(response, 'content') else str(response)
                    st.session_state.qa_pairs.append((question, answer))
                except Exception as e:
                    st.error(f"❌ Q&A Error: {e}")

        # Historique Q&A
        if st.session_state.qa_pairs:
            st.markdown("### 💬 Conversation History")
            for i, (q, a) in enumerate(reversed(st.session_state.qa_pairs), 1):
                with st.container(border=True):
                    st.markdown(f"**Q{i}:** {q}")
                    st.markdown(f"**A:** {a}")


if __name__ == "__main__":
    main()