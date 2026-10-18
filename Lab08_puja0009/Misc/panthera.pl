% FACTS

parent(panthera, palaeosinensis).
parent(panthera, node1).
parent(panthera, node4).

parent(node1, node2).
parent(node1, node3).

parent(node2, blytheae).
parent(node2, snow_leopard).

parent(node3, zdanskyi).
parent(node3, tiger).

parent(node4, node5).
parent(node4, node6).

parent(node5, gombaszoegensis).
parent(node5, jaguar).

parent(node6, leopard).
parent(node6, node7).

parent(node7, lion).
parent(node7, node8).

parent(node8, spelaea).
parent(node8, atrox).

% RULES

% Immediate ancestors
immediate_ancestors(X,Y) :-
	parent(A, X),
	parent(A, Y),
	X \= Y.

% Base case: direct descendant
descendant(X, Y) :-
    parent(Y, X).

% Recursive case: descendant through ancestors
descendant(X, Y) :-
    parent(Z, X),
    descendant(Z, Y).

% Indirect descendant only
indirect_descendant(X, Y) :-
    parent(Z, X),
    descendant(Z, Y).

% Grandparent (two generations above)
grandparent(X, Y) :-
    parent(X, Z),
    parent(Z, Y).