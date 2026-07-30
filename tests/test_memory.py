"""Unit tests for app.memory (ConversationMemory and SearchHistory)."""

from app.memory import ConversationMemory, SearchHistory


def test_conversation_memory_keeps_only_last_n_turns():
    memory = ConversationMemory(window_size=3)
    for i in range(5):
        memory.add_turn(f"question {i}", f"answer {i}")

    turns = memory.get_turns()
    assert len(turns) == 3
    assert [t.question for t in turns] == ["question 2", "question 3", "question 4"]


def test_conversation_memory_prompt_context_empty():
    memory = ConversationMemory(window_size=3)
    assert "No previous conversation" in memory.as_prompt_context()


def test_conversation_memory_prompt_context_contains_turns():
    memory = ConversationMemory(window_size=3)
    memory.add_turn("What are the side effects?", "Nausea and headache.")
    context = memory.as_prompt_context()
    assert "What are the side effects?" in context
    assert "Nausea and headache." in context


def test_conversation_memory_clear():
    memory = ConversationMemory(window_size=3)
    memory.add_turn("q", "a")
    memory.clear()
    assert memory.get_turns() == []


def test_search_history_avoids_consecutive_duplicates():
    history = SearchHistory()
    history.add("side effects")
    history.add("side effects")
    history.add("pregnancy")
    assert history.all() == ["side effects", "pregnancy"]


def test_search_history_allows_non_consecutive_repeat():
    history = SearchHistory()
    history.add("side effects")
    history.add("pregnancy")
    history.add("side effects")
    assert history.all() == ["side effects", "pregnancy", "side effects"]
