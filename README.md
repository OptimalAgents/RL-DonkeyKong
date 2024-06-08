# RL-DonkeyKong

## Notes

1. Można dać super dużą karę za bezsensowne skakanie
2. Dać mu nagrode za przejście w górę po drabinie. Zrobić to tak, że jak jest na drabinie i porusza się do góry to dostaje nagrodę, a jak porusza się w dół to dostaje karę.
   - Ze względu na to, że mamy gamme to będzie trzeba tą karę zrobić na tyle dużą, żeby warto było iść w górę. Kara > nagorda co do wartości bezwzględnej
3. Może warto zwiększyć szanse w losowaniu akcji na klknięcie w góre
   - może tylko wtedy gdy już epsilon spadnie do dolnej granicy?
   - może jakas heurystyka na to na który poziomie jesteśmy i w górą stronę chcemy iść?
4. Ewaluowac go po n-episodów, ale wtedy epsilon musi byc 0. Nie może być niedeterministyczny, chcemy mieć pewność, że to co się dzieje to wynik tego co model nauczył się do tej pory.
5. Dodać trackowanie eksperymentów przez TensorBoard
6. Może ustalić mu duża nagrodę za checkpointy, żeby szybciej się uczył?
