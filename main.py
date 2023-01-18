# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def day2_round_score(opponent, me):
    score = 0
    if me == 'X':
        score += 1
        if opponent == 'A':
            score += 3
        elif opponent == 'C':
            score += 6
    elif me == 'Y':
        score += 2
        if opponent == 'A':
            score += 6
        elif opponent == 'B':
            score += 3
    elif me == 'Z':
        score += 3
        if opponent == 'B':
            score += 6
        elif opponent == 'C':
            score += 3
    return score


def day2_what_to_play(opponent, result):
    if result == 'X':
        if opponent == 'A':
            return 'Z'
        elif opponent == 'B':
            return 'X'
        elif opponent == 'C':
            return 'Y'
    elif result == 'Y':
        if opponent == 'A':
            return 'X'
        elif opponent == 'B':
            return 'Y'
        elif opponent == 'C':
            return 'Z'
    elif result == 'Z':
        if opponent == 'A':
            return 'Y'
        elif opponent == 'B':
            return 'Z'
        elif opponent == 'C':
            return 'X'


def day2_part2():
    # Open the file in read mode
    with open('day2_input.txt', 'r') as file:
        # Read the entire contents of the file
        contents = file.read()

    # Split the contents into a list of lines
    lines = contents.split('\n')
    score = 0
    for line in lines:
        opponent = line.split()[0]
        result = line.split()[1]
        me = day2_what_to_play(opponent, result)
        score += day2_round_score(opponent, me)
    print(score)


def day2_part1():
    # Open the file in read mode
    with open('day2_input.txt', 'r') as file:
        # Read the entire contents of the file
        contents = file.read()

    # Split the contents into a list of lines
    lines = contents.split('\n')
    score = 0
    for line in lines:
        opponent = line.split()[0]
        me = line.split(' ')[1]
        score += day2_round_score(opponent, me)
    print(score)


def day1_part2():
    # Open the file in read mode
    with open('day1_input.txt', 'r') as file:
        # Read the entire contents of the file
        contents = file.read()

    # Split the contents into a list of lines
    lines = contents.split('\n')
    max_sum_1 = 0
    max_sum_2 = 0
    max_sum_3 = 0
    current_sum = 0
    for line in lines:
        if not line.isnumeric():
            if current_sum > max_sum_3:
                max_sum_3 = current_sum
                if max_sum_3 > max_sum_2:
                    temp = max_sum_2
                    max_sum_2 = max_sum_3
                    max_sum_3 = temp
                if max_sum_2 > max_sum_1:
                    temp = max_sum_1
                    max_sum_1 = max_sum_2
                    max_sum_2 = temp
            current_sum = 0

        else:
            current_sum += int(line)

    print(max_sum_1)
    print(max_sum_2)
    print(max_sum_3)
    print(max_sum_1 + max_sum_2 + max_sum_3)


def get_lines_from_file(filename):
    with open(filename, 'r') as file:
        # Read the entire contents of the file
        contents = file.read()

    # Split the contents into a list of lines
    lines = contents.split('\n')
    return lines


def get_priority(letter):
    all_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return all_letters.index(letter) + 1


def find_letter_in_two_strings(first_string, second_string):
    for letter in first_string:
        if letter in second_string:
            return letter


def find_letter_in_three_strings(first_string, second_string, third_string):
    for letter in first_string:
        if letter in second_string and letter in third_string:
            return letter


def day3_part2():
    lines = get_lines_from_file("day3_input.txt")
    sum_priorities = 0
    index = 0
    while index < len(lines):
        sum_priorities += get_priority(find_letter_in_three_strings(lines[index], lines[index + 1], lines[index + 2]))
        index += 3
    print(sum_priorities)


def day3_part1():
    lines = get_lines_from_file("day3_input.txt")
    sum_priorities = 0
    for line in lines:
        sum_priorities += get_priority(find_letter_in_two_strings(line[:len(line) // 2], line[len(line) // 2:]))

    print(sum_priorities)


def are_sections_contained(min_elf1, max_elf1, min_elf2, max_elf2):
    return (min_elf1 >= min_elf2 and max_elf1 <= max_elf2) or (min_elf2 >= min_elf1 and max_elf2 <= max_elf1)


def are_sections_overlapping(min_elf1, max_elf1, min_elf2, max_elf2):
    return (min_elf1 <= max_elf2) and (min_elf2 <= max_elf1)


def day4_part1():
    lines = get_lines_from_file("day4_input.txt")
    contains = 0
    for line in lines:
        elf1, elf2 = line.split(",")
        min_elf1, max_elf1 = elf1.split("-")
        min_elf2, max_elf2 = elf2.split("-")
        if are_sections_contained(int(min_elf1), int(max_elf1), int(min_elf2), int(max_elf2)):
            contains += 1
    print(contains)


def day4_part2():
    lines = get_lines_from_file("day4_input.txt")
    contains = 0
    for line in lines:
        elf1, elf2 = line.split(",")
        min_elf1, max_elf1 = elf1.split("-")
        min_elf2, max_elf2 = elf2.split("-")
        if are_sections_overlapping(int(min_elf1), int(max_elf1), int(min_elf2), int(max_elf2)):
            contains += 1
    print(contains)


def populate_stacks(lines):
    list_of_stacks = [[], [], [], [], [], [], [], [], []]
    for line in lines:
        index = 1
        while index < len(line):
            letter = line[index]
            if letter != ' ':
                list_of_stacks[int(index / 4)].append(letter)
            index += 4
    for stack in list_of_stacks:
        stack.reverse()
    return list_of_stacks


def day5_part1():
    lines = get_lines_from_file("day5_crates.txt")
    list_of_stacks = populate_stacks(lines)
    lines = get_lines_from_file("day5_procedure.txt")
    for line in lines:
        quantity, from_stack, to_stack = int(line.split()[1]), int(line.split()[3]), int(line.split()[5])
        for i in range(quantity):
            list_of_stacks[to_stack - 1].append(list_of_stacks[from_stack - 1].pop())
    print(''.join(stack.pop() for stack in list_of_stacks))


def day5_part2():
    lines = get_lines_from_file("day5_crates.txt")
    list_of_stacks = populate_stacks(lines)
    lines = get_lines_from_file("day5_procedure.txt")
    for line in lines:
        quantity, from_stack, to_stack = int(line.split()[1]), int(line.split()[3]), int(line.split()[5])
        temp_stack = []
        for i in range(quantity):
            temp_stack.append(list_of_stacks[from_stack - 1].pop())
        for i in range(quantity):
            list_of_stacks[to_stack - 1].append(temp_stack.pop())
    print(''.join(stack.pop() for stack in list_of_stacks))


def all_characters_different_in_array(array):
    for i in range(len(array) - 1):
        if array[i] in array[i + 1:]:
            return False
    return True


def day6_part1():
    line = get_lines_from_file("day6_input.txt")[0]
    array = ['*', '*', '*', '*']
    index_of_array = 0
    for i in range(len(line)):
        array[index_of_array] = line[i]
        if '*' not in array and all_characters_different_in_array(array):
            print(i + 1)
            exit(0)
        index_of_array = (index_of_array + 1) % 4
    print("Read entire input " + str(len(line)))


def day6_part2():
    line = get_lines_from_file("day6_input.txt")[0]
    array = ['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*']
    index_of_array = 0
    for i in range(len(line)):
        array[index_of_array] = line[i]
        if '*' not in array and all_characters_different_in_array(array):
            print(i + 1)
            exit(0)
        index_of_array = (index_of_array + 1) % 14
    print("Read entire input " + str(len(line)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    day6_part2()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
