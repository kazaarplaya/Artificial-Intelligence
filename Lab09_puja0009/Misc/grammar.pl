% grammar.pl - Knowledge base for simple English sentences using CFG

% Sentence structure
sentence --> noun_phrase, verb_phrase, preposition_phrase.

% Noun phrase structure
noun_phrase --> determiner, noun.
noun_phrase --> determiner, adjective, noun.

% Verb phrase structure
verb_phrase --> verb, noun_phrase.
verb_phrase --> verb.

% preposition phrase structure
preposition_phrase --> preposition, modifier, object.
preposition_phrase --> preposition, object.

% preposition
preposition --> [on].
preposition --> [in].
preposition --> [with].
preposition --> [under].

% Modifier
modifier --> [the].
modifier --> [my].

% Object 
object --> [table].
object --> [bed].
object --> [couch].

% Lexicon
determiner --> [the].
determiner --> [a].

noun --> [cat].
noun --> [dog].
noun --> [mouse].

adjective --> [big].
adjective --> [small].

verb --> [chases].
verb --> [sees].
verb --> [likes].