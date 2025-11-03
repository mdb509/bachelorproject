# Mastermind Rules

## Objective

* The **codebreaker** must guess the **secret code** created by the **codemaker** within a limited number of turns.
* After each guess, the codemaker provides feedback indicating how close the guess is to the secret code.

---

## Setup

* **Code length:** Usually 4 pegs (can vary from 3–6 or more).
* **Number of colors:** Typically 6 (e.g., Red, Green, Blue, Yellow, Orange, Purple).
* **Duplicates allowed:** Optional rule; determines whether the same color can appear more than once in the code.
* **Maximum attempts:** Usually 10–12 guesses per game.
* **Feedback type:** Black and white pegs (or numerical equivalents).

---

## Gameplay

1. **Secret Code Creation**

   * The codemaker chooses a hidden sequence of colors with a fixed length.
   * Duplicates are allowed or disallowed depending on the rules.

2. **Guessing**

   * The codebreaker submits a guess of the same length.
   * Each color position must be filled.

3. **Feedback**

   * For each guess, the codemaker provides feedback consisting of:

     * **Black pegs:** Correct color in the correct position.
     * **White pegs:** Correct color but in the wrong position.
   * Feedback order does **not** correspond to the position of the pegs.
   * Each color can only be counted once toward feedback (no double counting).

4. **Winning Condition**

   * The codebreaker wins if all pegs are black (exact match).
   * The codemaker wins if the codebreaker uses all attempts without finding the exact code.

5. **Game End**

   * The game ends immediately on a correct guess or after the final allowed attempt.
   * The secret code is revealed when the game ends.

---

## Optional Variants

* **Code length:** Can be adjusted for difficulty (e.g., 3–8).
* **Number of colors:** Can be increased (e.g., 8 or more).
* **No duplicates:** Only unique colors allowed.
* **No white feedback:** Feedback only for exact matches (hard mode).
* **Ordered feedback:** Feedback pegs correspond to specific positions (rare).
* **Reverse Mastermind:** The computer or codemaker tries to guess the player’s code.
* **Time-limited mode:** Each guess must be made within a fixed time period.

---

## Feedback Calculation (Algorithmic Rules)

1. Count all positions where `guess[i] == secret[i]` → black pegs.
2. Remove those matched positions from both sequences.
3. For the remaining colors, count matches by color only → white pegs.
4. Each color may only be matched once.

---

## Combinatorics

* Let **C** = number of colors, **N** = code length.

  * If duplicates are allowed → total codes = Cⁿ
  * If duplicates are not allowed → total codes = C! / (C − N)!

**Example:**
Standard game (6 colors, 4 positions)

* With duplicates: 6⁴ = 1,296 possible codes
* Without duplicates: 6 × 5 × 4 × 3 = 360 possible codes

---

Would you like me to make a **shorter, game-manual-style version** next (like something you’d put in a README or on a website)?
