% If-then-else with Operators and Rules
check_if_student_smart(X) :-
    ( X >= 85 -> 
        write('The student achieved a HD.')
    ; 
        write('The student did not achieve a HD.')
    ).

% If-then-else with Negation rule
raining(monday).
raining(sunday).

is_raining(Day) :-
    ( \+ raining(Day) ->
        write("It is not raining on this day.")
    ;
        write('It is raining on this day.')
    ).

% If-then-else that use input variables
check_if_broke :-
    write('Enter how much savings you have: '),
    read(Money),
    (Money >= 1000 ->
        write('You have alot of money for a uni student.')
    ;
        write('You are broke. This is normal for a uni student.')
    ).

% If-then-else with multiple branches
check_grade(Mark) :-
    (Mark >= 85 ->
        write('High Distinction')
    ; Mark >= 75 ->
        write('Distinction')
    ; Mark >= 65 ->
        write('Credit')
    ; Mark >= 50 ->
        write('Pass')
    ;
        write('Fail')
    ).

% Challenge question
if_statement(Condition, Then, Else) :-
    (call(Condition) ->
        call(Then)
    ;
        call(Else)
    ).

check_age(Age) :-
    if_statement(
        Age >= 18,
        write('You are old.'),
        write('You are young.')
    ).