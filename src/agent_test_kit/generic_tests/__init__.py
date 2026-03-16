"""
Ready-to-use generic test suites for any LLM agent.

Each class covers one testing category from the methodology.  To use them,
create a ``session`` pytest fixture in your ``conftest.py`` that returns an
initialized :class:`~agent_test_kit.AgentSession`, then inherit the desired
test class::

    from agent_test_kit.generic_tests import GenericEdgeCaseTests

    class TestEdgeCases(GenericEdgeCaseTests):
        pass

pytest will automatically discover and run all inherited test methods.

Готовые к использованию универсальные наборы тестов для любого LLM-агента.
Каждый класс охватывает одну категорию тестирования из методологии. Для
использования создайте фикстуру ``session`` в ``conftest.py``, возвращающую
инициализированный :class:`~agent_test_kit.AgentSession`, затем наследуйте
нужный тестовый класс. pytest автоматически обнаружит и запустит все
унаследованные тестовые методы.
"""

from agent_test_kit.generic_tests.edge_cases import GenericEdgeCaseTests
from agent_test_kit.generic_tests.security import GenericSecurityTests
from agent_test_kit.generic_tests.format import GenericFormatTests
from agent_test_kit.generic_tests.out_of_scope import GenericOutOfScopeTests
from agent_test_kit.generic_tests.memory import GenericMemoryTests
from agent_test_kit.generic_tests.stability import GenericStabilityTests
from agent_test_kit.generic_tests.performance import GenericPerformanceTests

__all__ = [
    "GenericEdgeCaseTests",
    "GenericSecurityTests",
    "GenericFormatTests",
    "GenericOutOfScopeTests",
    "GenericMemoryTests",
    "GenericStabilityTests",
    "GenericPerformanceTests",
]
