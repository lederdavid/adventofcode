# This is a sample Python script.
import numpy as np
from collections import defaultdict
import heapq as heap


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


def add_to_dir_size(stack_of_dirs, size):
    containing_dir = stack_of_dirs.pop()
    containing_dir[1] = containing_dir[1] + size
    stack_of_dirs.append(containing_dir)


def remove_dir_from_stack(stack_of_dirs, saved_dirs, condition, num):
    size = 0
    removed_dir = stack_of_dirs.pop()
    dir_size = int(removed_dir[1])
    if condition(dir_size, num):
        saved_dirs.append(removed_dir)
        size = dir_size
    add_to_dir_size(stack_of_dirs, dir_size)
    return size


def less_than(a, b):
    return a < b


def more_than(a, b):
    return a > b


def day7_part1():
    lines = get_lines_from_file("day7_input.txt")
    saved_dirs = []
    stack_of_dirs = []
    sum_of_dir_sizes = 0

    for line in lines:
        words = line.split()
        if words[0] == '$':  # command
            if words[1] == 'cd':
                current_dir = words[2]
                if current_dir != "..":
                    stack_of_dirs.append([current_dir, 0])
                else:
                    sum_of_dir_sizes += remove_dir_from_stack(stack_of_dirs, saved_dirs, less_than, 100000)

        elif words[0].isnumeric():  # regular file
            add_to_dir_size(stack_of_dirs, int(words[0]))

    while len(stack_of_dirs) > 1:
        sum_of_dir_sizes += remove_dir_from_stack(stack_of_dirs, saved_dirs, less_than, 100000)

    print(sum_of_dir_sizes)
    print(stack_of_dirs[0][1])


def find_smallest_dir(stack_of_dirs):
    min_size = 70000000
    min_dir = None
    for directory in stack_of_dirs:
        if int(directory[1]) < min_size:
            min_size = int(directory[1])
            min_dir = directory
    return min_dir


def day7_part2():
    lines = get_lines_from_file("day7_input.txt")
    saved_dirs = []
    stack_of_dirs = []

    for line in lines:
        words = line.split()
        if words[0] == '$':  # command
            if words[1] == 'cd':
                current_dir = words[2]
                if current_dir != "..":
                    stack_of_dirs.append([current_dir, 0])
                else:
                    remove_dir_from_stack(stack_of_dirs, saved_dirs, more_than, 2677139)

        elif words[0].isnumeric():  # regular file
            add_to_dir_size(stack_of_dirs, int(words[0]))

    while len(stack_of_dirs) > 1:
        remove_dir_from_stack(stack_of_dirs, saved_dirs, more_than, 2677139)

    print(find_smallest_dir(saved_dirs))


def larger_than_left_side(line, j):
    count = 0
    for i in reversed(range(j)):
        count += 1
        if int(line[i]) >= int(line[j]):
            return False, count
    return True, count


def larger_than_right_side(line, j):
    count = 0
    for i in range(j + 1, len(line)):
        count += 1
        if int(line[i]) >= int(line[j]):
            return False, count
    return True, count


def larger_than_up_side(lines, i, j):
    count = 0
    for index in reversed(range(i)):
        count += 1
        if int(lines[index][j]) >= int(lines[i][j]):
            return False, count
    return True, count


def larger_than_down_side(lines, i, j):
    count = 0
    for index in range(i + 1, len(lines)):
        count += 1
        if int(lines[index][j]) >= int(lines[i][j]):
            return False, count
    return True, count


def day8_part1():
    lines = get_lines_from_file("day8_input.txt")
    visible_trees = 2 * len(lines[0]) + 2 * (len(lines) - 2)
    for i in range(1, len(lines) - 1):
        for j in range(1, len(lines[i]) - 1):
            if larger_than_left_side(lines[i], j)[0] or \
                    larger_than_right_side(lines[i], j)[0] or \
                    larger_than_up_side(lines, i, j)[0] or \
                    larger_than_down_side(lines, i, j)[0]:
                visible_trees += 1
    print(visible_trees)


def day8_part2():
    lines = get_lines_from_file("day8_input.txt")
    max_score = 0
    for i in range(1, len(lines) - 1):
        for j in range(1, len(lines[i]) - 1):
            x = j
            current_score = larger_than_left_side(lines[i], j)[1] * \
                            larger_than_right_side(lines[i], j)[1] * \
                            larger_than_up_side(lines, i, j)[1] * \
                            larger_than_down_side(lines, i, j)[1]
            max_score = max(max_score, current_score)
    print(max_score)


def head_is_to_the_right(head, tail):
    return head[1] - tail[1] >= 1


def head_is_to_the_left(head, tail):
    return tail[1] - head[1] >= 1


def head_is_to_the_top(head, tail):
    return tail[0] - head[0] >= 1


def head_is_to_the_bottom(head, tail):
    return head[0] - tail[0] >= 1


def adjust_tail(head, tail):
    if head[0] - tail[0] > 1:
        tail[0] += 1
        if head_is_to_the_right(head, tail):
            tail[1] += 1
        elif head_is_to_the_left(head, tail):
            tail[1] -= 1
    elif tail[0] - head[0] > 1:
        tail[0] -= 1
        if head_is_to_the_right(head, tail):
            tail[1] += 1
        elif head_is_to_the_left(head, tail):
            tail[1] -= 1
    elif head[1] - tail[1] > 1:
        tail[1] += 1
        if head_is_to_the_top(head, tail):
            tail[0] -= 1
        elif head_is_to_the_bottom(head, tail):
            tail[0] += 1
    elif tail[1] - head[1] > 1:
        tail[1] -= 1
        if head_is_to_the_top(head, tail):
            tail[0] -= 1
        elif head_is_to_the_bottom(head, tail):
            tail[0] += 1


def day9_part1():
    lines = get_lines_from_file("day9_input.txt")
    visited = np.zeros((1000, 1000))
    head, tail = [500, 500], [500, 500]
    visited[500, 500] = 1
    for line in lines:
        direction, steps = line.split()
        for step in range(int(steps)):
            if direction == 'U':
                head[0] -= 1
            elif direction == 'D':
                head[0] += 1
            elif direction == 'L':
                head[1] -= 1
            else:
                head[1] += 1
            adjust_tail(head, tail)
            visited[tail[0], tail[1]] = 1
    print(np.count_nonzero(visited))


def day9_part2():
    lines = get_lines_from_file("day9_input.txt")
    visited = np.zeros((1000, 1000))
    head, tail = [500, 500], [[500, 500], [500, 500], [500, 500], [500, 500], [500, 500], [500, 500], [500, 500],
                              [500, 500], [500, 500]]
    visited[500, 500] = 1
    for line in lines:
        direction, steps = line.split()

        for step in range(int(steps)):
            if direction == 'U':
                head[0] -= 1
            elif direction == 'D':
                head[0] += 1
            elif direction == 'L':
                head[1] -= 1
            else:
                head[1] += 1
            adjust_tail(head, tail[0])

            for i in range(1, len(tail)):
                adjust_tail(tail[i - 1], tail[i])
            visited[tail[8][0], tail[8][1]] = 1

    print(np.count_nonzero(visited))


def add_cycle(cycle, x, sum_x):
    cycle += 1
    if (cycle + 20) % 40 == 0:
        sum_x += x * cycle
    return cycle, sum_x


def day10_part1():
    lines = get_lines_from_file("day10_input.txt")
    cycle = 0
    x = 1
    sum_x = 0
    for line in lines:
        cycle, sum_x = add_cycle(cycle, x, sum_x)
        if line.split()[0] == "addx":
            number = int(line.split()[1])
            cycle, sum_x = add_cycle(cycle, x, sum_x)
            x += number
    print(sum_x)


def add_cycle_and_draw(cycle, sprite_position, crt):
    cycle += 1
    row_position = cycle % 40
    if 0 <= row_position - sprite_position <= 2:
        crt += "#"
    else:
        crt += "."
    if row_position == 0:
        crt += "\n"
    return cycle, crt


def day10_part2():
    lines = get_lines_from_file("day10_input.txt")
    cycle = 0
    sprite_position = 1
    crt = ""
    for line in lines:
        cycle, crt = add_cycle_and_draw(cycle, sprite_position, crt)
        if line.split()[0] == "addx":
            number = int(line.split()[1])
            cycle, crt = add_cycle_and_draw(cycle, sprite_position, crt)
            sprite_position += number
    print(crt)


class Monkey:
    num_of_inspections = 0

    def __init__(self, items, operation, test):
        self.items = items
        self.operation = operation
        self.test = test

    def add_item(self, item):
        self.items.append(item)

    def inspect_item(self, modulo):
        self.num_of_inspections += 1
        item = self.items.pop(0)
        if "+" in self.operation:
            operand = self.operation.split("+")[1].strip()
            if operand == "old":
                worry_level = item + item
            else:
                worry_level = item + int(operand)
            if worry_level % self.test[0] == 0:
                return worry_level, self.test[1]
            else:
                return worry_level, self.test[2]  # returns tuple of (worry_level, monkey #)
        else:
            operand = self.operation.split("*")[1].strip()
            if operand == "old":
                worry_level = item * item
            else:
                worry_level = item * int(operand)
            if item % self.test[0] == 0:
                return worry_level % modulo, self.test[1]
            else:
                return worry_level % modulo, self.test[2]  # returns tuple of (worry_level, monkey #)

        # worry_level = int(worry_level / 3)  #part_1
        # if worry_level % self.test[0] == 0:
        #     return worry_level, self.test[1]
        # else:
        #     return worry_level, self.test[2]  # returns tuple of (worry_level, monkey #)


def process_monkey(lines):
    list_of_items = []
    for item in lines[1].split(":")[1].split(","):
        list_of_items.append(int(item.strip()))
    operation = lines[2].split(":")[1]
    test = (int(lines[3].split("by")[1].strip()), int(lines[4].split("monkey")[1].strip()),
            int(lines[5].split("monkey")[1].strip()))
    return Monkey(list_of_items, operation, test)


def do_round(array_of_monkeys, modulo):
    for i in range(len(array_of_monkeys)):
        monkey = array_of_monkeys[i]
        while len(monkey.items) > 0:
            monkey_throw = monkey.inspect_item(modulo)
            array_of_monkeys[monkey_throw[1]].add_item(monkey_throw[0])


def day11_part1():
    lines = get_lines_from_file("day11_input.txt")
    array_of_monkeys = [None, None, None, None, None, None, None, None]
    # lines = get_lines_from_file("day11_dummy.txt")
    # array_of_monkeys = [None, None, None, None]
    num_of_monkeys = len(array_of_monkeys)
    j = 0
    modulo = 1
    for i in range(num_of_monkeys):
        array_of_monkeys[i] = process_monkey(lines[j:j + 6])
        modulo *= array_of_monkeys[i].test[0]
        j += 7

    for i in range(10000):
        do_round(array_of_monkeys, modulo)

    max_inspections1 = 0
    max_inspections2 = 0
    for i in range(num_of_monkeys):
        inspections = array_of_monkeys[i].num_of_inspections
        if inspections > max_inspections1:
            max_inspections2 = max_inspections1
            max_inspections1 = inspections
        elif inspections > max_inspections2:
            max_inspections2 = inspections

    print(max_inspections1 * max_inspections2)


def dijkstra(input_matrix, starting_node):
    visited = set()
    parents_map = {}
    pq = []
    node_costs = defaultdict(lambda: float('inf'))
    node_costs[starting_node] = 0
    heap.heappush(pq, (0, starting_node))

    while pq:
        # go greedily by always extending the shorter cost nodes first
        _, node = heap.heappop(pq)
        visited.add(node)

        for adj_node in get_adjacent_nodes(input_matrix, node):
            if adj_node in visited:
                continue

            new_cost = node_costs[node] + 1
            if node_costs[adj_node] > new_cost:
                parents_map[adj_node] = node
                node_costs[adj_node] = new_cost
                heap.heappush(pq, (new_cost, adj_node))

    return parents_map, node_costs


def get_adjacent_nodes(input_matrix, current_position):
    result = []
    my_cell = input_matrix[current_position]
    if current_position[0] > 0:
        new_position = (current_position[0] - 1, current_position[1])
        new_cell = input_matrix[new_position]
        if new_cell != 0 and new_cell <= my_cell + 1:
            result.append(new_position)
    if current_position[0] < input_matrix.shape[0] - 1:
        new_position = (current_position[0] + 1, current_position[1])
        new_cell = input_matrix[new_position]
        if new_cell != 0 and new_cell <= my_cell + 1:
            result.append(new_position)
    if current_position[1] > 0:
        new_position = (current_position[0], current_position[1] - 1)
        new_cell = input_matrix[new_position]
        if new_cell != 0 and new_cell <= my_cell + 1:
            result.append(new_position)
    if current_position[1] < input_matrix.shape[1] - 1:
        new_position = (current_position[0], current_position[1] + 1)
        new_cell = input_matrix[new_position]
        if new_cell != 0 and new_cell <= my_cell + 1:
            result.append(new_position)
    return result


def get_elevation(letter):
    all_letters = "abcdefghijklmnopqrstuvwxyz"
    return all_letters.index(letter)


global_min = 100000000


def find_shortest_path(input_matrix, current_position, target_position):
    global global_min
    my_cell = input_matrix[current_position]
    my_matrix = input_matrix.copy()
    my_matrix[current_position] = 0
    # print(current_position)
    if current_position != target_position:
        if current_position[0] > 0:
            new_position = (current_position[0] - 1, current_position[1])
            new_cell = input_matrix[new_position]
            if new_cell != 0 and new_cell <= my_cell + 1:
                find_shortest_path(my_matrix, new_position, target_position)
        if current_position[0] < input_matrix.shape[0] - 1:
            new_position = (current_position[0] + 1, current_position[1])
            new_cell = input_matrix[new_position]
            if new_cell != 0 and new_cell <= my_cell + 1:
                find_shortest_path(my_matrix, new_position, target_position)
        if current_position[1] > 0:
            new_position = (current_position[0], current_position[1] - 1)
            new_cell = input_matrix[new_position]
            if new_cell != 0 and new_cell <= my_cell + 1:
                find_shortest_path(my_matrix, new_position, target_position)
        if current_position[1] < input_matrix.shape[1] - 1:
            new_position = (current_position[0], current_position[1] + 1)
            new_cell = input_matrix[new_position]
            if new_cell != 0 and new_cell <= my_cell + 1:
                find_shortest_path(my_matrix, new_position, target_position)
    else:
        global_min = min(global_min, input_matrix.size - np.count_nonzero(input_matrix))
        print(global_min)


def day12():
    lines = get_lines_from_file("day12_input.txt")
    # lines = get_lines_from_file("day12_dummy.txt")
    input_matrix = np.zeros((len(lines), len(lines[0])))
    i = 0
    current_position = (0, 0)
    target_position = (0, 0)
    for line in lines:
        j = 0
        for letter in line:
            if letter == "S":
                current_position = (i, j)
                input_matrix[i, j] = get_elevation("a")
            elif letter == "E":
                target_position = (i, j)
                input_matrix[i, j] = get_elevation("z")
            else:
                input_matrix[i, j] = get_elevation(letter)
            j += 1
        i += 1

    # part1:
    print(dijkstra(input_matrix, current_position)[1][target_position])

    # part2:
    min_steps = 10000000000000000
    for i in range(input_matrix.shape[0]):
        for j in range(input_matrix.shape[1]):
            if input_matrix[i, j] == 0:
                min_steps = min(min_steps, dijkstra(input_matrix, (i, j))[1][target_position])

    print(min_steps)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    day12()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
