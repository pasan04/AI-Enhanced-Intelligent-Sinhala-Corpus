# AI-Enhanced-Intelligent-Sinhala-Corpus

Sinhala is spoken by millions of people in
Sri Lanka, yet it remains a low-resource language for
Natural Language Processing (NLP) because relatively
few high-quality linguistic resources exist for it. The
corpora that do exist, including the well-known Sinmin
project, were built around storage schemas that only
support exact keyword matching or n-gram frequency
counts, so there is currently no easy way to look up
Sinhala documents by what they mean rather than
which words they contain. In this paper I build a
small working system that tries to close that gap:
an AI-enhanced Sinhala corpus that pairs a relational
document store with transformer-based sentence embeddings and vector similarity search. The corpus is
bootstrapped from NSina, currently th
