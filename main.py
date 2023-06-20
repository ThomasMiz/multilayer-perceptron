from src.network import Network
from src.theta import ThetaFunction, SimpleThetaFunction, LinealThetaFunction, TanhThetaFunction, LogisticThetaFunction

arch = [
    (2, SimpleThetaFunction),
    (1, SimpleThetaFunction)
]

n = Network(2, arch)
result = n.evaluate([1, 2])
print(result)