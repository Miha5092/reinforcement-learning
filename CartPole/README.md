# Cart Pole

This environment corresponds to the version of the cart-pole problem described by Barto and Sutton in their book Reinforcement Learning: An Introduction. A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

## Environment

The environment's description alongside documentation on how the environment library works can be found [gymnasium docs](https://gymnasium.farama.org/environments/classic_control/cart_pole/).

### Action Space

The action is a ndarray with shape `(1,)` which can take values `{0, 1}` indicating the direction of the fixed force the cart is pushed with.

-   0: Push the cart to the left
-   1: Push the cart to the right

### Observation Space

The observation is a ndarray with shape `(4,)` with the values corresponding to the following positions and velocities:

| | | | |
|-|-|-|-|
| Num | Observation | Min | Max |
| 0 | Cart Position | -2.4 | 2.4 |
| 1 | Cart Velocity | -Inf | Inf |
| 2 | Pole Angle | ~ -24° | ~ 24° |
| 3 | Pole Angular Velocity | -Inf | Inf |

### Rewards

Since the goal is to keep the pole upright for as long as possible, a reward of +1 for every step taken, including the termination step, is allotted. The threshold for rewards is 500 for v1 and 200 for v0.

## Agent