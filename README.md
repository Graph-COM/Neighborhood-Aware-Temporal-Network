# Neighborhood-Aware-Temporal-Network

Temporal networks have been widely used to model real-world complex systems
such as financial systems and e-commerce systems. In a temporal network, the joint
neighborhood of a set of nodes often provides crucial structural information on pre-
dicting whether they may interact at a certain time. However, recent representation
learning methods for temporal networks often fail to extract such information or de-
pend on extremely time-consuming feature construction approaches. To address the
issue, this work proposes Neighborhood-Aware Temporal network model (NAT).
For each node in the network, NAT abandons the commonly-used one-single-
vector-based representation while adopting a novel dictionary-type neighborhood
representation. Such a dictionary representation records a down-sampled set of the
neighboring nodes as keys, and allows fast construction of structural features for
a joint neighborhood of multiple nodes. We also design dedicated data structure
termed N-cache to support parallel access and update of those dictionary represen-
tations on GPUs. NAT gets evaluated over seven real-world large-scale temporal
networks. NAT not only outperforms all cutting-edge baselines by averaged 5.9%↑
and 6.0%↑ in transductive and inductive link prediction accuracy, respectively,
but also keeps scalable by achieving a speed-up of 4.1-76.7× against the base-
lines that adopts joint structural features and achieves a speed-up of 1.6-4.0×
against the baselines that cannot adopt those features
