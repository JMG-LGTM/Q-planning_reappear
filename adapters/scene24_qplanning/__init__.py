"""External interfaces for OpenPI and Scene24."""

from contextvars import ContextVar

# The vector runner uses a single environment. Its integer episode token also
# encodes the initial state, whereas the policy noise uses the decoded scene seed.
episode_policy_seed = ContextVar("scene24_episode_policy_seed", default=None)
