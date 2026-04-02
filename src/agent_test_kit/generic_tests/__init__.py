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
from agent_test_kit.generic_tests.prompt_security import GenericPromptSecurityTests
from agent_test_kit.generic_tests.social_engineering import GenericSocialEngineeringTests
from agent_test_kit.generic_tests.jailbreak import GenericJailbreakResistanceTests
from agent_test_kit.generic_tests.privacy import GenericPrivacyTests
from agent_test_kit.generic_tests.payload_safety import GenericPayloadSafetyTests
from agent_test_kit.generic_tests.format import GenericSurfaceFormatTests
from agent_test_kit.generic_tests.style import GenericStyleTests
from agent_test_kit.generic_tests.language import GenericLanguageTests
from agent_test_kit.generic_tests.off_topic import GenericOffTopicRefusalTests
from agent_test_kit.generic_tests.mixed_intent import GenericMixedIntentTests
from agent_test_kit.generic_tests.scope_recovery import GenericScopeRecoveryTests
from agent_test_kit.generic_tests.recall import GenericRecallTests
from agent_test_kit.generic_tests.corrections import GenericCorrectionTests
from agent_test_kit.generic_tests.long_context import GenericLongContextTests
from agent_test_kit.generic_tests.reproducibility import GenericReproducibilityTests
from agent_test_kit.generic_tests.paraphrase import GenericParaphraseConsistencyTests
from agent_test_kit.generic_tests.session_resilience import GenericSessionResilienceTests
from agent_test_kit.generic_tests.latency import GenericLatencyTests
from agent_test_kit.generic_tests.concurrency import GenericConcurrencyTests

__all__ = [
    "GenericEdgeCaseTests",
    "GenericPromptSecurityTests",
    "GenericSocialEngineeringTests",
    "GenericJailbreakResistanceTests",
    "GenericPrivacyTests",
    "GenericPayloadSafetyTests",
    "GenericSurfaceFormatTests",
    "GenericStyleTests",
    "GenericLanguageTests",
    "GenericOffTopicRefusalTests",
    "GenericMixedIntentTests",
    "GenericScopeRecoveryTests",
    "GenericRecallTests",
    "GenericCorrectionTests",
    "GenericLongContextTests",
    "GenericReproducibilityTests",
    "GenericParaphraseConsistencyTests",
    "GenericSessionResilienceTests",
    "GenericLatencyTests",
    "GenericConcurrencyTests",
]
