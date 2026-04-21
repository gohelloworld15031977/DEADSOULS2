"""
Демонстрация базового MAS.
Диалог Чичиков ↔ Тентетников без жёсткого FSM.
"""

from pathlib import Path

from src.agents.chichikov import ChichikovAgent
from src.agents.tentetnikov import TentetnikovAgent
from src.orchestrator import DialogueOrchestrator
from src.llm_wrapper import GogolLLMWrapper
from src.rag_wrapper import KnowledgeProvider


def main():
    """Главная функция демонстрации."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ BASIC MAS")
    print("Многоагентная система для реставрации второго тома «Мёртвых душ»")
    print("=" * 70)

    # 1. Инициализация LLM (Уровень 2)
    print("\n[1/5] Загрузка LLM...")
    llm = GogolLLMWrapper(model_path="gogol_finetuned_final")

    # 2. Инициализация RAG (Уровень 1)
    print("\n[2/5] Загрузка RAG...")
    try:
        rag = KnowledgeProvider(db_path="data/vector_db")
    except Exception as e:
        print(f"⚠️  RAG не загружен: {e}")
        rag = None

    # 3. Создание агентов (Уровень 3 - без FSM)
    print("\n[3/5] Создание агентов...")
    chichikov = ChichikovAgent()
    tentetnikov = TentetnikovAgent()

    # Инъекция LLM в агентов
    chichikov.set_llm_wrapper(llm)
    tentetnikov.set_llm_wrapper(llm)

    # 4. Настройка оркестратора
    print("\n[4/5] Настройка оркестратора...")
    orchestrator = DialogueOrchestrator()
    orchestrator.register_agent(chichikov)
    orchestrator.register_agent(tentetnikov)

    # Контекст сцены
    scene_context = """
    Поместье Тентетникова, утро 1830-х годов.
    Чичиков приехал с визитом. Тентетников встречает его в гостиной.
    Атмосфера: сдержанная, помещик не любит посетителей.
    За окном — русская усадьба, осень, листья опадают.
    """

    orchestrator.set_scene_context(scene_context)

    # 5. Запуск диалога
    print("\n[5/5] Запуск диалога...")
    print("\n" + "=" * 70)
    print("ДИАЛОГ: Чичиков ↔ Тентетников")
    print("=" * 70)

    dialogue = orchestrator.run_dialogue(
        agent_a_name="Чичиков",
        agent_b_name="Тентетников",
        num_turns=3,
        scene_context=scene_context
    )

    # Вывод диалога
    print("\n")
    for line in dialogue:
        print(line)
        print()

    # Сводка состояний
    print("=" * 70)
    print("СВОДКА СОСТОЯНИЙ")
    print("=" * 70)
    print(f"Чичиков: {chichikov.get_state_summary()}")
    print(f"Тентетников: {tentetnikov.get_state_summary()}")

    # Сохранение
    print("\n" + "=" * 70)
    print("СОХРАНЕНИЕ ДИАЛОГА")
    print("=" * 70)
    orchestrator.save_dialogue(str(Path(__file__).parent / "dialogue_output.txt"), dialogue)
    print("Диалог сохранён в: dialogue_output.txt")

    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 70)


if __name__ == "__main__":
    main()
