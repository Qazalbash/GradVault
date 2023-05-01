def guessing_game(secret_number, falto):
    count = 0
    while count < 5:
        count += 1
        num = int(input())
        print(f"Attempt number {count}")
        if num < secret_number:
            if check(count):
                print("Sorry, you lose! Too many wrong guesses.")
            else:
                print("Try again! Your guess is too low.")
        elif num > secret_number:
            if check(count):
                print("Sorry, you lose! Too many wrong guesses.")
            else:
                print("Try again! Your guess is too high.")
        else:
            print(
                f"Congratulations, you won! You guessed the secret number {secret_number} in {count} guesses."
            )
            break


def check(count):
    if count == 5:
        return ("Sorry, you lose! Too many wrong guesses.")


secret_number = int(input())
guessing_game(secret_number, 1)
