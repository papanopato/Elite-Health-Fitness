from agents.orchestrator import Orchestrator

def main():
    # Initialisation de notre chef d'orchestre
    chef = Orchestrator()
    
    print("--- Elite Health & Fitness Agent ---")
    print("Tapez 'quitter' pour sortir.")
    
    while True:
        user_input = input("\nVous : ")
        if user_input.lower() == "quitter":
            break
            
        # On envoie la demande à l'orchestrateur
        reponse = chef.route_query(user_input)
        print(f"Agent : {reponse}")

if __name__ == "__main__":
    main()
