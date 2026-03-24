"""
functions.py
Полный набор функций для работы со свободными узлами.
Содержит:
- Базовые операции (каноническая форма, сдвиги, индексы)
- Движения Рейдемейстера (Ω1, Ω2, Ω3) с вариантами
- Анализ чётности и несократимости
- Генерация диаграмм и классификация
- Инварианты (скобка чётности, z-инвариант Гибсона)
- Визуализация и трассировка
"""

import itertools
import copy
import time
from collections import Counter, deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================================
# БАЗОВЫЕ ФУНКЦИИ: каноническая форма, сдвиги, индексы
# ============================================================================

def renumerate_filter(diagrams):
    """
    Фильтрует список диаграмм, оставляя только уникальные, т.е. те,
    которые не могут быть получены друг из друга с помощью
    - Перенумерации хорд
    - Циклического сдвига
    - Зеркального отражения

    Параметры:
    ----------
    diagrams : list of list of int
        Список хордовых диаграмм для фильтрации

    Возвращает:
    -----------
    list of list of int
        Список уникальных диаграмм
    """
    if not diagrams:
        return []

    unique_diagrams = []
    canonical_forms_seen = []

    for diagram in diagrams:
        if _is_canonical_form_unique(diagram, canonical_forms_seen):
            unique_diagrams.append(diagram)

            canonical_form = _compute_canonical_form(diagram)
            canonical_forms_seen.append(canonical_form)

    return unique_diagrams


def _compute_canonical_form(diagram):
    """
    Вычисляет каноническую форму диаграммы для сравнения.

    Возвращает отсортированный список списков индексов для каждой хорды,
    что инвариантно относительно перенумерации, сдвига и отражения.
    """
    chord_indices = []
    for chord_label in set(diagram):
        indices = get_indices(diagram, chord_label)
        chord_indices.append(sorted(indices))

    return sorted(chord_indices)


def _is_canonical_form_unique(diagram, existing_forms):
    """
    Проверяет, является ли диаграмма уникальной относительно существующих форм.
    """
    variants_to_check = _generate_all_equivalent_variants(diagram)

    for variant in variants_to_check:
        variant_form = _compute_canonical_form(variant)
        if variant_form in existing_forms:
            return False

    return True


def _generate_all_equivalent_variants(diagram):
    """
    Генерирует все диаграммы, эквивалентные данной (сдвиги + отражения).
    """
    variants = []
    n = len(diagram)

    # Все циклические сдвиги оригинальной диаграммы
    for shift_amount in range(n):
        shifted = shift(diagram, shift_amount)
        variants.append(shifted)

    # Все циклические сдвиги отраженной диаграммы
    reflected = diagram[::-1]
    for shift_amount in range(n):
        shifted_reflected = shift(reflected, shift_amount)
        variants.append(shifted_reflected)

    return variants


def get_indices(diagram, chord_label):
    """
    Находит все позиции (индексы) заданной хорды в диаграмме.

    Каждая хорда в диаграмме свободного узла представлена двумя концами,
    поэтому для корректной хорды функция всегда возвращает список из двух индексов.

    Параметры:
    ----------
    diagram : list of int
        Хордовая диаграмма, представленная списком меток хорд
    chord_label : int
        Метка хорды, для которой ищутся позиции

    Возвращает:
    -----------
    list of int
        Список индексов в диаграмме, где встречается заданная хорда.
        Для корректной диаграммы всегда содержит ровно два элемента.

    Примеры:
    --------
    >>> get_indices([1, 2, 1, 3, 2, 3], 1)
    [0, 2]
    >>> get_indices([1, 2, 1, 3, 2, 3], 2)
    [1, 4]
    """
    return [index for index, chord in enumerate(diagram) if chord == chord_label]


def shift(diagram, steps):
    """
    Циклически сдвигает хордовую диаграмму на заданное число позиций.

    Диаграмма свободного узла представляет собой циклическую последовательность,
    поэтому сдвиг не меняет топологические свойства узла, а лишь изменяет
    начальную точку обхода диаграммы.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма
    steps : int
        Количество позиций для сдвига:
        - steps > 0: сдвиг вправо
        - steps < 0: сдвиг влево
        - steps = 0: диаграмма без изменений

    Возвращает:
    -----------
    list of int
        Сдвинутая диаграмма той же длины, что и исходная

    Примеры:
    --------
    >>> shift([1, 2, 1, 3, 2, 3], 1)
    [3, 1, 2, 1, 3, 2]
    >>> shift([1, 2, 1, 3, 2, 3], -1)
    [2, 1, 3, 2, 3, 1]
    """
    if not diagram or steps == 0:
        return diagram.copy()

    steps = steps % len(diagram)

    if steps < 0:
        steps = len(diagram) + steps

    return diagram[-steps:] + diagram[:-steps]

def diagram_info(diagram, timeout=30):
    """
    Выводит полную информацию о хордовой диаграмме свободного узла.
    
    Параметры:
    ----------
    diagram : list of int
        Хордовая диаграмма
    timeout : int
        Время на поиск упрощения в секундах
    """
    print("=" * 80)
    print(f"ДИАГРАММА: {diagram}")
    print("=" * 80)
    
    # 1. Число хорд
    chords = sorted(set(diagram))
    num_chords = len(chords)
    print(f"\n1. ОСНОВНЫЕ ХАРАКТЕРИСТИКИ:")
    print(f"   Число хорд: {num_chords}")
    print(f"   Длина диаграммы: {len(diagram)} точек")
    
    # 2. Четность хорд
    print(f"\n2. ЧЕТНОСТЬ ХОРД:")
    even_chords_list = []
    odd_chords_list = []
    for chord in chords:
        parity = get_parity(diagram, chord)
        if parity == 0:
            even_chords_list.append(chord)
            print(f"   Хорда {chord}: ЧЕТНАЯ")
        else:
            odd_chords_list.append(chord)
            print(f"   Хорда {chord}: НЕЧЕТНАЯ")
    
    print(f"\n   Итого: четных {len(even_chords_list)}, нечетных {len(odd_chords_list)}")
    
    # 3. Несократимость
    irreducible = is_diagram_irreducible(diagram)
    print(f"\n3. НЕСОКРАТИМОСТЬ:")
    print(f"   Диаграмма {'НЕСОКРАТИМА' if irreducible else 'СОКРАТИМА'}")
    
    # 4. Каноническая форма
    normalized = normalize_chord_diagram(diagram)
    canonical_form = find_canonical_cyclic_form(normalized)
    print(f"\n4. КАНОНИЧЕСКАЯ ФОРМА:")
    print(f"   {canonical_form}")
    
    # 5. Упрощаемость уменьшающими движениями
    easing = is_directly_easing(diagram)
    print(f"\n5. УПРОЩАЕМОСТЬ:")
    if easing:
        print(f"   Диаграмма УПРОЩАЕТСЯ уменьшающими движениями")
    else:
        print(f"   Диаграмма НЕ УПРОЩАЕТСЯ уменьшающими движениями")
    
    # 6. Скобка четности
    print(f"\n6. СКОБКА ЧЕТНОСТИ:")
    bracket = compute_parity_bracket(diagram, verbose=False)
    if bracket:
        print(f"   Слагаемых: {len(bracket)}")
        for i, term in enumerate(bracket, 1):
            term_str = str(term)
            if len(term_str) > 80:
                term_str = term_str[:80] + "..."
            print(f"     {i}: {term_str}")
    else:
        print(f"   Пустая скобка (все слагаемые нулевые)")
    
    # 7. Инвариант Гибсона (только для четного числа хорд)
    if num_chords % 2 == 0:
        print(f"\n7. ИНВАРИАНТ ГИБСОНА:")
        gibson_invariant_terms(diagram)
    
    # 8. Визуализация диаграммы
    print(f"\n8. ВИЗУАЛИЗАЦИЯ ДИАГРАММЫ:")
    draw_chord_diagram(diagram, size=300)
    plt.show()
    
    # 9. Визуализация упрощения (если диаграмма упрощается)
    if easing and len(diagram) > 0:
        print(f"\n9. ВИЗУАЛИЗАЦИЯ УПРОЩЕНИЯ:")
        print(f"   (поиск пути упрощения, таймаут {timeout} сек)")
        path = trace_simplification_with_chords(diagram, timeout=timeout)
        if path and len(path) > 1:
            fig = visualize_simplification(path, max_cols=6, diagram_size=100)
            plt.show()
        else:
            print("   Не удалось найти путь упрощения")
    
    print("\n" + "=" * 80)


# ============================================================================
# ПРОВЕРКА ПРИМЕНИМОСТИ ДВИЖЕНИЙ РЕЙДЕМЕЙСТЕРА
# ============================================================================

def is_first_reidemeister_applicable(diagram):
    """
    Проверяет, применимо ли первое движение Рейдемейстера к диаграмме.

    Первое движение Рейдемейстера применимо, если в диаграмме существует
    изолированная хорда - то есть два конца одной хорды расположены
    в соседних позициях циклической диаграммы.

    Возвращает:
    -----------
    bool
        True если первое движение применимо, False в противном случае.
    """
    diagram_length = len(diagram)

    for position in range(diagram_length):
        current_chord = diagram[position]
        next_chord = diagram[(position + 1) % diagram_length]

        if current_chord == next_chord:
            return True

    return False


def is_second_reidemeister_applicable(diagram):
    """
    Проверяет, применимо ли второе движение Рейдемейстера к диаграмме.

    Возвращает:
    -----------
    bool
        True если второе движение применимо, False в противном случае.
    """
    diagram_length = len(diagram)

    for i in range(diagram_length - 2):
        chord_a = diagram[i]
        chord_b = diagram[i - 1]

        if i == 0:
            for j in range(i + 2, diagram_length - 1):
                chord_c = diagram[j]
                chord_d = diagram[j - 1]

                if (chord_c == chord_a or chord_c == chord_b) and \
                   (chord_d == chord_a or chord_d == chord_b):
                    return True
        else:
            for j in range(i + 2, diagram_length):
                chord_c = diagram[j]
                chord_d = diagram[j - 1]

                if (chord_c == chord_a or chord_c == chord_b) and \
                   (chord_d == chord_a or chord_d == chord_b):
                    return True

    return False


def is_third_reidemeister_applicable(diagram):
    """
    Проверяет, применимо ли третье движение Рейдемейстера к диаграмме.

    Возвращает:
    -----------
    bool
        True если третье движение применимо, False в противном случае.
    """
    working_diagram = diagram.copy()
    diagram_length = len(working_diagram)

    for shift_count in range(diagram_length):
        if _check_third_move_types(working_diagram):
            return True

        working_diagram = shift(working_diagram, 1)

    return False


def _check_third_move_types(diagram):
    """
    Проверяет все четыре типа третьего движения Рейдемейстера для данной диаграммы.
    """
    chord_indices = _get_unique_chord_indices(diagram)

    if not chord_indices:
        return False

    base_chord_indices = chord_indices[0]
    x1, x2 = base_chord_indices

    # Тип 1->2
    for position in range(len(diagram)):
        if ([x1 + 1, position - 1] in chord_indices and
            [x2 - 1, position] in chord_indices):
            return True

    # Тип 2->1
    for position in range(len(diagram)):
        if ([x1 + 1, position - 1] in chord_indices and
            [position, x2 - 1] in chord_indices):
            return True

    # Тип 3->4
    for position in range(len(diagram)):
        if ([x1 + 1, position - 1] in chord_indices and
            [x2 + 1, position] in chord_indices):
            return True

    # Тип 4->3
    for position in range(len(diagram)):
        if ([x1 + 1, position + 1] in chord_indices and
            [position, x2 - 1] in chord_indices):
            return True

    return False


def _get_unique_chord_indices(diagram):
    """
    Возвращает отсортированный список уникальных пар индексов для всех хорд.
    """
    indices_set = set()

    for chord_label in diagram:
        chord_positions = get_indices(diagram, chord_label)
        indices_set.add(tuple(sorted(chord_positions)))

    unique_indices = [list(position_tuple) for position_tuple in indices_set]
    return sorted(unique_indices)

# ============================================================================
# ДВИЖЕНИЯ РЕЙДЕМЕЙСТЕРА (основные функции)
# ============================================================================

def mod_1(diagram):
    """
    Применяет первое уменьшающее движение Рейдемейстера ко всем возможным позициям.

    Возвращает все диаграммы, которые могут быть получены удалением изолированных хорд
    (пар соседних концов одной хорды).

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of list of int
        Список всех уникальных диаграмм после применения первого движения
    """
    result_diagrams = []

    for position in range(len(diagram)):
        current_chord = diagram[position]
        previous_chord = diagram[position - 1]

        if current_chord == previous_chord:
            modified_diagram = diagram.copy()

            if position == 0:
                del modified_diagram[-1]
                del modified_diagram[0]
            else:
                del modified_diagram[position - 1]
                del modified_diagram[position - 1]

            result_diagrams.append(modified_diagram)

    return renumerate_filter(result_diagrams)


def mod_1_increasing(diagram):
    """
    Применяет первое увеличивающее движение Рейдемейстера - добавляет изолированную хорду.

    Добавляет новую хорду таким образом, что ее концы располагаются в соседних позициях
    в циклической диаграмме.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of list of int
        Список всех уникальных диаграмм после применения первого увеличивающего движения
    """
    if not diagram:
        return [[1, 1]]

    new_chord_label = max(diagram) + 1
    generated_diagrams = []
    n = len(diagram)

    # Оба конца подряд в существующих промежутках
    for i in range(n + 1):
        new_diagram = diagram[:i] + [new_chord_label, new_chord_label] + diagram[i:]
        generated_diagrams.append(new_diagram)

    # Циклический случай: один конец в начале, другой в конце
    wrap_around = [new_chord_label] + diagram + [new_chord_label]
    generated_diagrams.append(wrap_around)

    return renumerate_filter(generated_diagrams)


def mod_2_non_reducing(diagram):
    """
    Применяет второе неуменьшающее движение Рейдемейстера.

    Второе неуменьшающее движение меняет местами концы двух хорд,
    не уменьшая общее количество хорд в диаграмме.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of list of int
        Список всех уникальных диаграмм после применения второго неуменьшающего движения
    """
    result_diagrams = [diagram.copy()]

    for i in range(len(diagram) - 2):
        chord_a = diagram[i]
        chord_b = diagram[i - 1]

        if i == 0:
            for j in range(i + 2, len(diagram) - 1):
                chord_c = diagram[j]
                chord_d = diagram[j - 1]

                if (chord_c == chord_a and chord_d == chord_b) or \
                   (chord_c == chord_b and chord_d == chord_a):
                    modified_diagram = diagram.copy()
                    modified_diagram[i] = chord_b
                    modified_diagram[i - 1] = chord_a
                    result_diagrams.append(modified_diagram)
        else:
            for j in range(i + 2, len(diagram)):
                chord_c = diagram[j]
                chord_d = diagram[j - 1]

                if (chord_c == chord_a and chord_d == chord_b) or \
                   (chord_c == chord_b and chord_d == chord_a):
                    modified_diagram = diagram.copy()
                    modified_diagram[i] = chord_b
                    modified_diagram[i - 1] = chord_a
                    result_diagrams.append(modified_diagram)

    return renumerate_filter(result_diagrams)


def mod_2_reducing(diagram):
    """
    Применяет второе уменьшающее движение Рейдемейстера.

    Второе уменьшающее движение удаляет две хорды, которые могут быть
    сокращены попарно, уменьшая общее количество хорд в диаграмме.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of list of int
        Список всех уникальных диаграмм после применения второго уменьшающего движения
    """
    result_diagrams = []

    for i in range(len(diagram) - 2):
        chord_a = diagram[i]
        chord_b = diagram[i - 1]

        if i == 0:
            for j in range(i + 2, len(diagram) - 1):
                chord_c = diagram[j]
                chord_d = diagram[j - 1]

                if (chord_c == chord_a and chord_d == chord_b) or \
                   (chord_c == chord_b and chord_d == chord_a):
                    modified_diagram = diagram.copy()
                    positions_to_remove = sorted([i - 1, i, j - 1, j], reverse=True)
                    for pos in positions_to_remove:
                        del modified_diagram[pos]
                    result_diagrams.append(modified_diagram)
        else:
            for j in range(i + 2, len(diagram)):
                chord_c = diagram[j]
                chord_d = diagram[j - 1]

                if (chord_c == chord_a and chord_d == chord_b) or \
                   (chord_c == chord_b and chord_d == chord_a):
                    modified_diagram = diagram.copy()
                    positions_to_remove = sorted([i - 1, i, j - 1, j], reverse=True)
                    for pos in positions_to_remove:
                        del modified_diagram[pos]
                    result_diagrams.append(modified_diagram)

    return renumerate_filter(result_diagrams)


def mod_2_increasing(diagram):
    """
    Применяет второе увеличивающее движение Рейдемейстера - добавляет две хорды,
    которые могут быть сокращены вторым движением.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of list of int
        Список всех уникальных диаграмм после применения второго увеличивающего движения
    """
    if not diagram:
        return [[1, 2, 1, 2]]

    new_chord_a = max(diagram) + 1
    new_chord_b = new_chord_a + 1
    generated_diagrams = []
    n = len(diagram)

    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            # Вариант 1: обе пары в прямом порядке [A,B] и [A,B]
            diagram1 = _insert_chord_pair(diagram, i, j, new_chord_a, new_chord_b,
                                          new_chord_a, new_chord_b)
            generated_diagrams.append(diagram1)

            # Вариант 2: первая пара прямая [A,B], вторая обратная [B,A]
            diagram2 = _insert_chord_pair(diagram, i, j, new_chord_a, new_chord_b,
                                          new_chord_b, new_chord_a)
            generated_diagrams.append(diagram2)

    generated_diagrams.extend(_generate_cyclic_variants(diagram, new_chord_a, new_chord_b))

    return renumerate_filter(generated_diagrams)


def _insert_chord_pair(base_diagram, pos1, pos2, chord1a, chord1b, chord2a, chord2b):
    """
    Вставляет две пары хорд в указанные позиции.
    """
    diagram = base_diagram.copy()

    # Вставляем первую пару
    diagram[pos1:pos1] = [chord1a, chord1b]

    # Корректируем позицию для второй пары (учитываем уже вставленные элементы)
    adjusted_pos2 = pos2 + 2 if pos2 >= pos1 else pos2

    # Вставляем вторую пару
    diagram[adjusted_pos2:adjusted_pos2] = [chord2a, chord2b]

    return diagram


def _generate_cyclic_variants(diagram, chord_a, chord_b):
    """
    Генерирует варианты, где пары хорд разрываются циклически (через границу начала/конца).
    """
    variants = []
    n = len(diagram)

    for i in range(n + 1):
        # Вариант: первая пара разорвана [A, ..., B]
        variant1 = [chord_a] + diagram[i:] + diagram[:i] + [chord_b]
        # Добавляем вторую пару в различных позициях
        for j in range(1, len(variant1)):
            if j != 0 and j != len(variant1) - 1:
                final_variant = variant1[:j] + [chord_a, chord_b] + variant1[j:]
                variants.append(final_variant)

                final_variant_rev = variant1[:j] + [chord_b, chord_a] + variant1[j:]
                variants.append(final_variant_rev)

    return variants


def mod_2(diagram):
    """
    Применяет второе движение Рейдемейстера (все варианты).

    Возвращает список из трех элементов:
    - Первый элемент: все диаграммы после второго неуменьшающего движения
    - Второй элемент: все диаграммы после второго уменьшающего движения
    - Третий элемент: все диаграммы после второго увеличивающего движения

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of [list, list, list]
        [неуменьшающие_диаграммы, уменьшающие_диаграммы, увеличивающие_диаграммы]
    """
    non_reducing = mod_2_non_reducing(diagram)
    reducing = mod_2_reducing(diagram)
    increasing = mod_2_increasing(diagram)

    return [non_reducing, reducing, increasing]


def mod_3(diagram):
    """
    Применяет третье движение Рейдемейстера ко всем возможным позициям.

    Возвращает все диаграммы, которые могут быть получены применением
    третьего движения к исходной диаграмме.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of list of int
        Список всех уникальных диаграмм после применения третьего движения
    """
    result_diagrams = [diagram.copy()]
    working_diagram = diagram.copy()
    diagram_length = len(working_diagram)

    for shift_count in range(diagram_length):
        chord_indices = _get_sorted_chord_indices(working_diagram)

        if chord_indices:
            base_chord = chord_indices[0]
            x1, x2 = base_chord

            _check_type_1_to_2(working_diagram, x1, x2, chord_indices, result_diagrams)
            _check_type_2_to_1(working_diagram, x1, x2, chord_indices, result_diagrams)
            _check_type_3_to_4(working_diagram, x1, x2, chord_indices, result_diagrams)
            _check_type_4_to_3(working_diagram, x1, x2, chord_indices, result_diagrams)

        working_diagram = shift(working_diagram, 1)

    return renumerate_filter(result_diagrams)


def _get_sorted_chord_indices(diagram):
    """
    Возвращает отсортированный список уникальных пар индексов для всех хорд.
    """
    indices_set = set()

    for chord_label in diagram:
        chord_positions = get_indices(diagram, chord_label)
        indices_set.add(tuple(sorted(chord_positions)))

    unique_indices = [list(position_tuple) for position_tuple in indices_set]
    return sorted(unique_indices)


def _check_type_1_to_2(diagram, x1, x2, chord_indices, result_diagrams):
    """Проверяет третье движение типа 1->2."""
    for position in range(len(diagram)):
        condition_1 = [x1 + 1, position - 1] in chord_indices
        condition_2 = [x2 - 1, position] in chord_indices
        condition_3 = ([x1, x2] != [x1 + 1, position - 1])
        condition_4 = ([x1, x2] != [x2 - 1, position])
        condition_5 = ([x1 + 1, position - 1] != [x2 - 1, position])

        if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
            new_diagram = diagram.copy()
            new_diagram[x1] = diagram[x1 + 1]
            new_diagram[x1 + 1] = diagram[x1]
            new_diagram[x2 - 1] = diagram[x2]
            new_diagram[x2] = diagram[x2 - 1]
            new_diagram[position] = diagram[position - 1]
            new_diagram[position - 1] = diagram[position]
            result_diagrams.append(new_diagram)


def _check_type_2_to_1(diagram, x1, x2, chord_indices, result_diagrams):
    """Проверяет третье движение типа 2->1."""
    for position in range(len(diagram)):
        condition_1 = [x1 + 1, position - 1] in chord_indices
        condition_2 = [position, x2 - 1] in chord_indices
        condition_3 = ([x1, x2] != [x1 + 1, position - 1])
        condition_4 = ([x1, x2] != [position, x2 - 1])
        condition_5 = ([x1 + 1, position - 1] != [position, x2 - 1])

        if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
            new_diagram = diagram.copy()
            new_diagram[x1] = diagram[x1 + 1]
            new_diagram[x1 + 1] = diagram[x1]
            new_diagram[x2 - 1] = diagram[x2]
            new_diagram[x2] = diagram[x2 - 1]
            new_diagram[position] = diagram[position - 1]
            new_diagram[position - 1] = diagram[position]
            result_diagrams.append(new_diagram)


def _check_type_3_to_4(diagram, x1, x2, chord_indices, result_diagrams):
    """Проверяет третье движение типа 3->4."""
    for position in range(len(diagram)):
        condition_1 = [x1 + 1, position - 1] in chord_indices
        condition_2 = [x2 + 1, position] in chord_indices
        condition_3 = ([x1, x2] != [x1 + 1, position - 1])
        condition_4 = ([x1, x2] != [x2 + 1, position])
        condition_5 = ([x1 + 1, position - 1] != [x2 + 1, position])

        if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
            new_diagram = diagram.copy()
            new_diagram[x1] = diagram[x1 + 1]
            new_diagram[x1 + 1] = diagram[x1]
            new_diagram[x2 + 1] = diagram[x2]
            new_diagram[x2] = diagram[x2 + 1]
            new_diagram[position] = diagram[position - 1]
            new_diagram[position - 1] = diagram[position]
            result_diagrams.append(new_diagram)


def _check_type_4_to_3(diagram, x1, x2, chord_indices, result_diagrams):
    """Проверяет третье движение типа 4->3."""
    for position in range(len(diagram)):
        condition_1 = [x1 + 1, position + 1] in chord_indices
        condition_2 = [position, x2 - 1] in chord_indices
        condition_3 = ([x1, x2] != [x1 + 1, position + 1])
        condition_4 = ([x1, x2] != [position, x2 - 1])
        condition_5 = ([x1 + 1, position + 1] != [position, x2 - 1])

        if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
            new_diagram = diagram.copy()
            new_diagram[x1] = diagram[x1 + 1]
            new_diagram[x1 + 1] = diagram[x1]
            new_diagram[x2 - 1] = diagram[x2]
            new_diagram[x2] = diagram[x2 - 1]
            new_diagram[position] = diagram[position + 1]
            new_diagram[position + 1] = diagram[position]
            result_diagrams.append(new_diagram)

# ============================================================================
# АНАЛИЗ ЧЁТНОСТИ И НЕСОКРАТИМОСТИ
# ============================================================================

def is_engaged(diagram, chord_a, chord_b):
    """
    Проверяет, зацеплены ли две хорды в диаграмме свободного узла.

    Две хорды считаются зацепленными, если концы одной хорды расположены
    по разные стороны от концов другой хорды в циклической диаграмме.

    Параметры:
    ----------
    diagram : list of int
        Хордовая диаграмма
    chord_a : int
        Метка первой хорды
    chord_b : int
        Метка второй хорды

    Возвращает:
    -----------
    bool
        True если хорды зацеплены, False если не зацеплены

    Примеры:
    --------
    >>> is_engaged([1, 2, 1, 2], 1, 2)
    True
    >>> is_engaged([1, 1, 2, 2], 1, 2)
    False
    """
    indices_a = get_indices(diagram, chord_a)
    indices_b = get_indices(diagram, chord_b)

    a_start, a_end = sorted(indices_a)
    b_start, b_end = sorted(indices_b)

    b_first_inside = a_start < b_start < a_end
    b_second_inside = a_start < b_end < a_end

    return b_first_inside != b_second_inside


def get_parity(diagram, chord):
    """
    Вычисляет четность хорды в диаграмме свободного узла.

    Параметры:
    ----------
    diagram : list of int
        Хордовая диаграмма
    chord : int
        Метка хорды

    Возвращает:
    -----------
    int
        1 если хорда нечетная (пересекает нечетное число других хорд),
        0 если хорда четная (пересекает четное число других хорд)
    """
    indices = get_indices(diagram, chord)
    start, end = sorted(indices)

    inner_chords = {}

    for position in range(start + 1, end):
        current_chord = diagram[position]
        if current_chord in inner_chords:
            del inner_chords[current_chord]
        else:
            inner_chords[current_chord] = 1

    intersection_count = len(inner_chords)
    return intersection_count % 2


def is_diagram_uneven(diagram):
    """
    Проверяет, является ли диаграмма нечетной.

    Диаграмма считается нечетной, если все ее хорды имеют нечетную четность.

    Параметры:
    ----------
    diagram : list of int
        Хордовая диаграмма для проверки

    Возвращает:
    -----------
    bool
        True если все хорды нечетные, False если найдена хотя бы одна четная хорда

    Примеры:
    --------
    >>> is_diagram_uneven([1, 2, 1, 2])
    True
    >>> is_diagram_uneven([1, 2, 3, 1, 3, 2])
    False
    """
    for chord_label in set(diagram):
        if get_parity(diagram, chord_label) == 0:
            return False
    return True


def is_diagram_irreducible(diagram):
    """
    Проверяет, является ли диаграмма несократимой.

    Диаграмма считается несократимой, если для любой пары различных хорд
    найдется третья хорда, которая зацеплена ровно с одной из них.

    Параметры:
    ----------
    diagram : list of int
        Хордовая диаграмма для проверки

    Возвращает:
    -----------
    bool
        True если диаграмма несократима, False если сократима

    Примеры:
    --------
    >>> is_diagram_irreducible([1, 2, 1, 2])
    True
    >>> is_diagram_irreducible([1, 2, 3, 1, 2, 3])
    False
    """
    chord_labels = list(set(diagram))
    chord_pairs = list(itertools.combinations(chord_labels, 2))

    for chord_a, chord_b in chord_pairs:
        remaining_chords = copy.copy(chord_labels)
        remaining_chords.remove(chord_a)
        remaining_chords.remove(chord_b)

        distinguishing_chord_found = False
        for chord_c in remaining_chords:
            if is_engaged(diagram, chord_a, chord_c) != is_engaged(diagram, chord_b, chord_c):
                distinguishing_chord_found = True
                break

        if not distinguishing_chord_found:
            return False

    return True


# ============================================================================
# ГЕНЕРАЦИЯ ДИАГРАММ
# ============================================================================

def add_chord(diagram):
    """
    Генерирует все уникальные диаграммы, полученные добавлением одной хорды.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of list of int
        Список уникальных диаграмм с добавленной хордой
    """
    if not diagram:
        return [[1, 1]]

    new_chord_label = max(diagram) + 1
    generated_diagrams = []

    for first_pos in range(len(diagram) + 1):
        diagram_with_first_end = (diagram[:first_pos] +
                                 [new_chord_label] +
                                 diagram[first_pos:])

        for second_pos in range(len(diagram_with_first_end) + 1):
            final_diagram = (diagram_with_first_end[:second_pos] +
                            [new_chord_label] +
                            diagram_with_first_end[second_pos:])

            generated_diagrams.append(final_diagram)

    return renumerate_filter(generated_diagrams)


# ============================================================================
# УПРОЩЕНИЕ И ЭКВИВАЛЕНТНОСТЬ
# ============================================================================

def is_directly_easing(diagram):
    """
    Проверяет, можно ли упростить диаграмму, применяя только уменьшающие движения.

    Использует комбинацию поиска эквивалентных диаграмм через 2-е и 3-е движения
    с одновременной проверкой возможности применения 1-го и 2-го уменьшающих движений.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    bool
        True если диаграмму можно упростить, False если нельзя
    """
    if len(diagram) <= 8:
        return True

    non_reducing_2 = mod_2_non_reducing(diagram)
    non_reducing_3 = mod_3(diagram)

    equivalent_diagrams = non_reducing_2 + non_reducing_3 + [diagram]
    equivalent_diagrams = renumerate_filter(equivalent_diagrams)

    continue_expansion = True
    current_2 = non_reducing_2
    current_3 = non_reducing_3

    while continue_expansion:
        continue_expansion = False
        new_2_diagrams = []
        new_3_diagrams = []

        for diagram_2 in current_2:
            equivalent_3 = mod_3(diagram_2)
            for eq_diag in equivalent_3:
                test_set = equivalent_diagrams + [eq_diag]
                if len(equivalent_diagrams) < len(renumerate_filter(test_set)):
                    equivalent_diagrams.append(eq_diag)
                    new_3_diagrams.append(eq_diag)
                    continue_expansion = True

        for diagram_2 in current_2:
            equivalent_2 = mod_2_non_reducing(diagram_2)
            for eq_diag in equivalent_2:
                test_set = equivalent_diagrams + [eq_diag]
                if len(equivalent_diagrams) < len(renumerate_filter(test_set)):
                    equivalent_diagrams.append(eq_diag)
                    new_2_diagrams.append(eq_diag)
                    continue_expansion = True

        for diagram_3 in current_3:
            equivalent_2 = mod_2_non_reducing(diagram_3)
            for eq_diag in equivalent_2:
                test_set = equivalent_diagrams + [eq_diag]
                if len(equivalent_diagrams) < len(renumerate_filter(test_set)):
                    equivalent_diagrams.append(eq_diag)
                    new_2_diagrams.append(eq_diag)
                    continue_expansion = True

        for diagram_3 in current_3:
            equivalent_3 = mod_3(diagram_3)
            for eq_diag in equivalent_3:
                test_set = equivalent_diagrams + [eq_diag]
                if len(equivalent_diagrams) < len(renumerate_filter(test_set)):
                    equivalent_diagrams.append(eq_diag)
                    new_3_diagrams.append(eq_diag)
                    continue_expansion = True

        all_current_diagrams = renumerate_filter(
            new_3_diagrams + new_2_diagrams + current_2 + current_3
        )

        if _check_reduction_possibility(all_current_diagrams):
            return True

        if continue_expansion:
            current_2 = new_2_diagrams.copy()
            current_3 = new_3_diagrams.copy()
        else:
            break

    return is_directly_easing_2(diagram)


def _check_reduction_possibility(diagrams):
    """
    Проверяет возможность применения уменьшающих движений к набору диаграмм.
    """
    current_diagrams = diagrams

    while True:
        reduced_diagrams = []

        for diagram in current_diagrams:
            if is_first_reidemeister_applicable(diagram):
                reduced_diagrams.extend(mod_1(diagram))

            if is_second_reidemeister_applicable(diagram):
                reduced_diagrams.extend(mod_2_reducing(diagram))

        reduced_diagrams = renumerate_filter(reduced_diagrams)

        for diagram in reduced_diagrams:
            if len(diagram) <= 8:
                return True

        if len(reduced_diagrams) == 0:
            break

        current_diagrams = reduced_diagrams

    return False


def is_directly_easing_2(diagram):
    """
    Резервный алгоритм проверки упрощаемости диаграммы.
    Сначала вычисляет все эквивалентные диаграммы, затем пытается их упростить.
    """
    if len(diagram) <= 8:
        return True

    equivalent = get_equivalent(diagram)
    simplified = False
    should_break = False

    while not simplified:
        reduced_diagrams = []

        for eq_diagram in equivalent:
            if is_first_reidemeister_applicable(eq_diagram):
                reduced_diagrams.extend(mod_1(eq_diagram))

            if is_second_reidemeister_applicable(eq_diagram):
                reduced_diagrams.extend(mod_2_reducing(eq_diagram))

        reduced_diagrams = renumerate_filter(reduced_diagrams)
        equivalent = []

        for reduced_diagram in reduced_diagrams:
            if len(reduced_diagram) <= 8:
                simplified = True
                should_break = True
                break
            else:
                equivalent.extend(get_equivalent(reduced_diagram))
                equivalent = renumerate_filter(equivalent)

        if len(reduced_diagrams) == 0:
            should_break = True

        if should_break:
            break

    return simplified


def get_equivalent(diagram):
    """
    Вычисляет все диаграммы, эквивалентные данной по 2-му и 3-му неуменьшающим движениям.

    Алгоритм рекурсивно расширяет множество эквивалентных диаграмм, применяя
    вторые и третьи движения Рейдемейстера до тех пор, пока не будет достигнуто
    замкнутое множество.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of list of int
        Список всех уникальных диаграмм, эквивалентных исходной
    """
    equivalent_2 = mod_2_non_reducing(diagram)
    equivalent_3 = mod_3(diagram)

    equivalent_diagrams = renumerate_filter(equivalent_2 + equivalent_3)

    current_2_diagrams = equivalent_2
    current_3_diagrams = equivalent_3

    while True:
        can_expand = False
        new_2_diagrams = []
        new_3_diagrams = []

        for diagram_2 in current_2_diagrams:
            additional_3 = mod_3(diagram_2)
            for new_diagram in additional_3:
                test_set = equivalent_diagrams + [new_diagram]
                if len(equivalent_diagrams) < len(renumerate_filter(test_set)):
                    equivalent_diagrams.append(new_diagram)
                    new_3_diagrams.append(new_diagram)
                    can_expand = True

        for diagram_2 in current_2_diagrams:
            additional_2 = mod_2_non_reducing(diagram_2)
            for new_diagram in additional_2:
                test_set = equivalent_diagrams + [new_diagram]
                if len(equivalent_diagrams) < len(renumerate_filter(test_set)):
                    equivalent_diagrams.append(new_diagram)
                    new_2_diagrams.append(new_diagram)
                    can_expand = True

        for diagram_3 in current_3_diagrams:
            additional_2 = mod_2_non_reducing(diagram_3)
            for new_diagram in additional_2:
                test_set = equivalent_diagrams + [new_diagram]
                if len(equivalent_diagrams) < len(renumerate_filter(test_set)):
                    equivalent_diagrams.append(new_diagram)
                    new_2_diagrams.append(new_diagram)
                    can_expand = True

        for diagram_3 in current_3_diagrams:
            additional_3 = mod_3(diagram_3)
            for new_diagram in additional_3:
                test_set = equivalent_diagrams + [new_diagram]
                if len(equivalent_diagrams) < len(renumerate_filter(test_set)):
                    equivalent_diagrams.append(new_diagram)
                    new_3_diagrams.append(new_diagram)
                    can_expand = True

        if not can_expand:
            break

        current_2_diagrams = new_2_diagrams.copy()
        current_3_diagrams = new_3_diagrams.copy()

    return renumerate_filter(equivalent_diagrams)


def simplify_with_equivalence_search(diagram, timeout=30, max_depth=10, verbose=False,
                                      _start_time=None, _current_depth=0, _visited=None):
    """
    Упрощает диаграмму через поиск в пространстве эквивалентных диаграмм (DFS).

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма
    timeout : int
        Максимальное время выполнения в секундах
    max_depth : int
        Максимальная глубина рекурсии
    verbose : bool
        Если True, выводит подробные логи
    _start_time, _current_depth, _visited : внутренние параметры для рекурсии

    Возвращает:
    -----------
    list of int
        Упрощенная диаграмма (наименьшая достигнутая)
    """
    if _start_time is None:
        _start_time = time.time()
    if _visited is None:
        _visited = set()

    if len(diagram) <= 2:
        if verbose and len(diagram) == 0:
            print("Достигнут тривиальный узел!")
        return diagram

    if _current_depth >= max_depth:
        if verbose:
            print(f"Достигнута максимальная глубина {max_depth}")
        return diagram

    if time.time() - _start_time > timeout:
        if verbose:
            print(f"Время вышло ({timeout} сек)")
        return diagram

    diagram_tuple = tuple(diagram)
    if diagram_tuple in _visited:
        if verbose:
            print(f"Обнаружен цикл, глубина {_current_depth}")
        return diagram
    _visited.add(diagram_tuple)

    if verbose:
        print(f"{'  ' * _current_depth}Глубина {_current_depth}: диаграмма {diagram} ({len(diagram)} точек)")

    directly_simplified = _apply_direct_reductions_until_stable(
        diagram, _start_time, timeout, verbose, _current_depth
    )

    if len(directly_simplified) <= 2:
        if verbose and len(directly_simplified) == 0:
            print(f"{'  ' * _current_depth}Прямое упрощение до тривиального узла!")
        return directly_simplified

    best_result = directly_simplified
    best_size = len(directly_simplified)

    try:
        if verbose:
            print(f"{'  ' * _current_depth}Поиск эквивалентных диаграмм...")

        equivalent_diagrams = get_equivalent(directly_simplified)

        if verbose:
            print(f"{'  ' * _current_depth}Найдено {len(equivalent_diagrams)} эквивалентных диаграмм")

        for i, eq_diagram in enumerate(equivalent_diagrams):
            if time.time() - _start_time > timeout:
                break

            if len(eq_diagram) > best_size + 4:
                continue

            if verbose:
                print(f"{'  ' * _current_depth}Обрабатываем эквивалентную {i+1}/{len(equivalent_diagrams)}: {eq_diagram}")

            candidate = simplify_with_equivalence_search(
                eq_diagram, timeout, max_depth, verbose,
                _start_time, _current_depth + 1, _visited
            )

            if len(candidate) < best_size:
                best_result = candidate
                best_size = len(candidate)

                if verbose:
                    print(f"{'  ' * _current_depth}УЛУЧШЕНИЕ: {len(directly_simplified)} точек -> {best_size} точек")

                if best_size <= 2:
                    return best_result

    except Exception as e:
        if verbose:
            print(f"{'  ' * _current_depth}Ошибка в get_equivalent: {e}")

    return best_result


def _apply_direct_reductions_until_stable(diagram, start_time, timeout, verbose, depth):
    """
    Применяет 1-е и 2-е уменьшающие движения пока есть прогресс.
    """
    current = diagram.copy()
    changed = True
    iteration = 0

    while changed:
        if time.time() - start_time > timeout:
            break

        changed = False
        previous_size = len(current)
        iteration += 1

        if is_first_reidemeister_applicable(current):
            reduced_1 = mod_1(current)
            if reduced_1 and len(reduced_1[0]) < len(current):
                old_diagram = current.copy()
                current = reduced_1[0]
                changed = True
                if verbose:
                    print(f"{'  ' * depth}1-е движение: {old_diagram} -> {current}")
                continue

        if is_second_reidemeister_applicable(current):
            reduced_2 = mod_2_reducing(current)
            if reduced_2:
                smallest = min(reduced_2, key=len)
                if len(smallest) < len(current):
                    old_diagram = current.copy()
                    current = smallest
                    changed = True
                    if verbose:
                        print(f"{'  ' * depth}2-е движение: {old_diagram} -> {current}")
                    continue

        if len(current) >= previous_size:
            if verbose and iteration > 1:
                print(f"{'  ' * depth}Прямое упрощение завершено")
            break

    return current


def deep_equivalence_simplify(diagram, timeout=60, verbose=False):
    """
    BFS-версия упрощения диаграммы.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма
    timeout : int
        Максимальное время выполнения в секундах
    verbose : bool
        Если True, выводит подробные логи

    Возвращает:
    -----------
    list of int
        Упрощенная диаграмма (наименьшая достигнутая)
    """
    start_time = time.time()
    best_result = diagram
    best_size = len(diagram)
    visited = set()
    queue = deque([(diagram, 0)])

    if verbose:
        print("Запуск BFS-упрощения...")
        print(f"Начальная диаграмма: {diagram} ({len(diagram)} точек)")

    while queue and (time.time() - start_time) < timeout:
        current_diag, depth = queue.popleft()

        current_tuple = tuple(current_diag)
        if current_tuple in visited:
            continue
        visited.add(current_tuple)

        if verbose:
            print(f"Глубина {depth}: обрабатываем {current_diag} ({len(current_diag)} точек)")

        simplified = _apply_direct_reductions_until_stable_bfs(
            current_diag, start_time, timeout, verbose, depth
        )

        if len(simplified) < best_size:
            old_best = best_result
            best_result = simplified
            best_size = len(simplified)

            if verbose:
                print(f"УЛУЧШЕНИЕ: {len(current_diag)} точек -> {best_size} точек (глубина {depth})")

            if best_size <= 2:
                if verbose:
                    print("Достигнут тривиальный узел!")
                return best_result

        if len(simplified) > 2 and depth < 5:
            try:
                equivalents = get_equivalent(simplified)
                if verbose:
                    print(f"  Найдено {len(equivalents)} эквивалентных диаграмм")

                for i, eq in enumerate(equivalents):
                    if (time.time() - start_time) < timeout and tuple(eq) not in visited:
                        if verbose:
                            print(f"  Добавляем в очередь: {eq} ({len(eq)} точек)")
                        queue.append((eq, depth + 1))
            except Exception as e:
                if verbose:
                    print(f"  Ошибка в get_equivalent: {e}")
                continue

    if verbose:
        print(f"BFS-упрощение завершено. Лучший результат: {best_result} ({best_size} точек)")

    return best_result


def _apply_direct_reductions_until_stable_bfs(diagram, start_time, timeout, verbose, depth):
    """
    Упрощение для BFS-версии (без рекурсивных вызовов).
    """
    current = diagram.copy()
    changed = True

    while changed:
        if time.time() - start_time > timeout:
            break

        changed = False
        previous_size = len(current)

        if is_first_reidemeister_applicable(current):
            reduced_1 = mod_1(current)
            if reduced_1 and len(reduced_1[0]) < len(current):
                current = reduced_1[0]
                changed = True
                if verbose:
                    print(f"  Глубина {depth}: 1-е движение -> {current}")
                continue

        if is_second_reidemeister_applicable(current):
            reduced_2 = mod_2_reducing(current)
            if reduced_2:
                smallest = min(reduced_2, key=len)
                if len(smallest) < len(current):
                    current = smallest
                    changed = True
                    if verbose:
                        print(f"  Глубина {depth}: 2-е движение -> {current}")
                    continue

        if len(current) >= previous_size:
            break

    return current


# ============================================================================
# СКОБКА ЧЁТНОСТИ - ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def even_chord_quantity(diagram):
    """
    Вычисляет количество четных хорд в диаграмме свободного узла.

    Четная хорда - это хорда, которая пересекает четное число других хорд.

    Параметры:
    ----------
    diagram : list of int
        Хордовая диаграмма

    Возвращает:
    -----------
    int
        Количество четных хорд в диаграмме

    Примеры:
    --------
    >>> even_chord_quantity([1, 2, 1, 2])
    0
    >>> even_chord_quantity([1, 2, 3, 1, 3, 2])
    1
    """
    if not diagram:
        return 0

    unique_chords = list(set(diagram))
    even_count = 0

    for chord in unique_chords:
        if get_parity(diagram, chord) == 0:
            even_count += 1

    return even_count


def identify_even_chords(diagram):
    """
    Идентифицирует четные хорды в хордовой диаграмме.

    Хорда называется четной, если она пересекается с четным числом
    других хорд диаграммы.

    Параметры:
    ----------
    diagram : list
        Хордовая диаграмма, представленная списком меток

    Возвращает:
    -----------
    list
        Список меток четных хорд
    """
    chord_labels = list(set(diagram))
    even_chords = []

    for chord in chord_labels:
        chord_positions = get_indices(diagram, chord)
        crossing_count = 0

        for other_chord in chord_labels:
            if other_chord == chord:
                continue
            other_positions = get_indices(diagram, other_chord)
            if (chord_positions[0] < other_positions[0] < chord_positions[1] < other_positions[1] or
                other_positions[0] < chord_positions[0] < other_positions[1] < chord_positions[1]):
                crossing_count += 1

        if crossing_count % 2 == 0:
            even_chords.append(chord)

    return even_chords


def normalize_chord_diagram(diagram):
    """
    Приводит хордовую диаграмму к нормализованной форме.

    Нормализация заключается в перенумерации хорд в лексикографическом
    порядке, начиная с 1.

    Параметры:
    ----------
    diagram : list
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list
        Нормализованная хордовая диаграмма
    """
    if not diagram:
        return []

    unique_chords = sorted(set(diagram))
    normalization_map = {original: normalized + 1 for normalized, original in enumerate(unique_chords)}

    normalized_diagram = [normalization_map[chord] for chord in diagram]
    return normalized_diagram


def construct_resolution_diagram(original_diagram, even_chords, resolution_scheme):
    """
    Строит техническую диаграмму для заданной схемы разрешений четных хорд.

    Для каждой четной хорды применяется одно из двух разрешений:
    - 'break': меняет порядок обхода концов хорды
    - 'flip': меняет направление обхода

    Параметры:
    ----------
    original_diagram : list
        Исходная хордовая диаграмма
    even_chords : list
        Список четных хорд
    resolution_scheme : tuple
        Кортеж, задающий тип разрешения для каждой четной хорды

    Возвращает:
    -----------
    list
        Техническая диаграмма с примененными разрешениями
    """
    resolution_diagram = []
    resolution_mapping = dict(zip(even_chords, resolution_scheme))

    for chord in original_diagram:
        if chord in even_chords:
            resolution_type = resolution_mapping[chord]

            if resolution_type == 'break':
                appearances_count = sum(1 for x in resolution_diagram if abs(x) == chord)
                if appearances_count == 0:
                    resolution_diagram.extend([chord, -chord])
                elif appearances_count == 2:
                    resolution_diagram.extend([-chord, chord])
            else:  # resolution_type == 'flip'
                appearances_count = sum(1 for x in resolution_diagram if abs(x) == chord)
                if appearances_count == 0:
                    resolution_diagram.extend([chord, -chord])
                elif appearances_count == 2:
                    resolution_diagram.extend([chord, -chord])
        else:
            resolution_diagram.append(chord)

    return resolution_diagram


def extract_connected_components(resolution_diagram, even_chords, resolution_scheme, verbose=False):
    """
    Извлекает компоненты связности из технической диаграммы.

    Компоненты связности соответствуют траекториям обхода диаграммы
    с учетом примененных разрешений четных хорд.

    Параметры:
    ----------
    resolution_diagram : list
        Техническая диаграмма с разрешениями
    even_chords : list
        Список четных хорд
    resolution_scheme : tuple
        Схема разрешений, примененная к диаграмме
    verbose : bool
        Если True, выводит подробные логи вычислений

    Возвращает:
    -----------
    list
        Список компонент связности, каждая из которых представлена
        списком меток хорд
    """
    diagram_length = len(resolution_diagram)
    visited_positions = set()
    connected_components = []
    resolution_mapping = dict(zip(even_chords, resolution_scheme))

    for start_index in range(diagram_length):
        if start_index in visited_positions:
            continue

        if verbose:
            print('=' * 60)
            print(f"Начало новой компоненты с индекса {start_index}")

        current_index = start_index
        traversal_direction = 1
        current_component = []
        component_visited = set()

        start_conditions = (start_index, traversal_direction)
        first_iteration = True

        while True:
            if not first_iteration and (current_index, traversal_direction) == start_conditions:
                if verbose:
                    print(f"Компонента завершена: вернулись в начало {current_index} с направлением {traversal_direction}")
                break

            first_iteration = False

            if current_index in component_visited:
                if verbose:
                    print(f"Обнаружено повторное посещение индекса {current_index}")
                break

            component_visited.add(current_index)
            visited_positions.add(current_index)
            current_value = resolution_diagram[current_index]

            if verbose:
                print(f'Посещенный индекс: {current_index}, метка хорды: {current_value}, направление: {traversal_direction}')

            if current_value > 0 and abs(current_value) not in even_chords:
                current_component.append(current_value)

            teleportation_occurred = False
            if abs(current_value) in even_chords:
                if traversal_direction == 1:
                    next_index = (current_index + 1) % diagram_length
                else:
                    next_index = (current_index - 1) % diagram_length

                next_value = resolution_diagram[next_index]

                if abs(next_value) == abs(current_value) and next_value != current_value:
                    resolution_type = resolution_mapping[abs(current_value)]

                    if resolution_type == 'flip':
                        traversal_direction *= -1

                    pair_indices = [idx for idx, val in enumerate(resolution_diagram)
                                  if val == current_value and idx != current_index and idx != next_index]

                    if pair_indices:
                        pair_index = pair_indices[0]

                        component_visited.add(current_index)
                        visited_positions.add(current_index)

                        if verbose:
                            print(f'Телепортация на индекс: {pair_index}, метка хорды: {resolution_diagram[pair_index]}, новое направление: {traversal_direction}')

                        current_index = pair_index
                        component_visited.add(current_index)
                        visited_positions.add(current_index)

                        if (current_index, traversal_direction) == start_conditions:
                            if verbose:
                                print(f"Компонента завершена после телепортации: вернулись в начало {current_index} с направлением {traversal_direction}")
                            break

                        if traversal_direction == 1:
                            current_index = (current_index + 1) % diagram_length
                        else:
                            current_index = (current_index - 1) % diagram_length

                        teleportation_occurred = True
                        continue

            if not teleportation_occurred:
                if traversal_direction == 1:
                    current_index = (current_index + 1) % diagram_length
                else:
                    current_index = (current_index - 1) % diagram_length

        connected_components.append(current_component)
        if verbose:
            print(f"Добавлена компонента: {current_component}")

    return connected_components


def normalize_single_component(component):
    """
    Нормализует одну компоненту связности.

    Находит лексикографически минимальное представление компоненты
    среди всех циклических сдвигов и отражений.

    Параметры:
    ----------
    component : list
        Исходная компонента связности

    Возвращает:
    -----------
    list
        Нормализованная компонента связности
    """
    if not component:
        return []

    cyclic_representations = []
    component_length = len(component)

    for shift in range(component_length):
        shifted_component = component[shift:] + component[:shift]
        cyclic_representations.append(shifted_component)

    reflected_component = component[::-1]
    for shift in range(component_length):
        shifted_reflection = reflected_component[shift:] + reflected_component[:shift]
        cyclic_representations.append(shifted_reflection)

    canonical_representation = min(cyclic_representations)
    return canonical_representation


def simplify_by_reidemeister_two(components, verbose=False):
    """
    Применяет второе движение Рейдемейстера для упрощения компонент.

    Алгоритм рекурсивно удаляет пары хорд, образующие второе движение,
    до тех пор, пока это возможно.

    Параметры:
    ----------
    components : list
        Список компонент связности для упрощения
    verbose : bool
        Если True, выводит подробные логи

    Возвращает:
    -----------
    list
        Упрощенный список компонент связности
    """
    if not components:
        return [[]]

    current_components = [comp.copy() for comp in components]
    simplification_occurred = True

    while simplification_occurred:
        simplification_occurred = False

        chord_endpoints = {}
        for comp_index, component in enumerate(current_components):
            for position, chord in enumerate(component):
                if chord not in chord_endpoints:
                    chord_endpoints[chord] = []
                chord_endpoints[chord].append((comp_index, position))

        chords_with_two_ends = [chord for chord, endpoints in chord_endpoints.items()
                              if len(endpoints) == 2]

        for first_index in range(len(chords_with_two_ends)):
            for second_index in range(first_index + 1, len(chords_with_two_ends)):
                chord_a, chord_b = chords_with_two_ends[first_index], chords_with_two_ends[second_index]
                endpoints_a = chord_endpoints[chord_a]
                endpoints_b = chord_endpoints[chord_b]

                adjacent_pairs = []

                comp_endpoints_a = {}
                comp_endpoints_b = {}
                for comp_idx, pos in endpoints_a:
                    if comp_idx not in comp_endpoints_a:
                        comp_endpoints_a[comp_idx] = []
                    comp_endpoints_a[comp_idx].append(pos)
                for comp_idx, pos in endpoints_b:
                    if comp_idx not in comp_endpoints_b:
                        comp_endpoints_b[comp_idx] = []
                    comp_endpoints_b[comp_idx].append(pos)

                for comp_idx in set(comp_endpoints_a.keys()) & set(comp_endpoints_b.keys()):
                    component = current_components[comp_idx]
                    comp_length = len(component)

                    for pos_a in comp_endpoints_a[comp_idx]:
                        for pos_b in comp_endpoints_b[comp_idx]:
                            if (pos_a + 1) % comp_length == pos_b or (pos_b + 1) % comp_length == pos_a:
                                adjacent_pairs.append((comp_idx, pos_a, pos_b))

                if len(adjacent_pairs) >= 2:
                    for pair1_index in range(len(adjacent_pairs)):
                        for pair2_index in range(pair1_index + 1, len(adjacent_pairs)):
                            comp1, pos1a, pos1b = adjacent_pairs[pair1_index]
                            comp2, pos2a, pos2b = adjacent_pairs[pair2_index]

                            used_ends_pair1 = {(comp1, pos1a), (comp1, pos1b)}
                            used_ends_pair2 = {(comp2, pos2a), (comp2, pos2b)}

                            if used_ends_pair1.isdisjoint(used_ends_pair2):
                                simplified_components = []
                                for idx, component in enumerate(current_components):
                                    filtered_component = [chord for chord in component
                                                        if chord not in [chord_a, chord_b]]
                                    simplified_components.append(filtered_component)

                                current_components = simplified_components
                                simplification_occurred = True
                                if verbose:
                                    print(f"  Удалены хорды {chord_a} и {chord_b}")
                                break

                        if simplification_occurred:
                            break
                    if simplification_occurred:
                        break
                if simplification_occurred:
                    break
            if simplification_occurred:
                break

    return current_components


def is_null_term(components):
    """
    Проверяет, является ли слагаемое нулевым.

    Слагаемое считается нулевым, если оно содержит ≥2 компонент,
    из которых хотя бы одна тривиальна (пустая).

    Параметры:
    ----------
    components : list
        Список компонент связности

    Возвращает:
    -----------
    bool
        True если слагаемое нулевое, иначе False
    """
    if len(components) > 1 and any(len(comp) == 0 for comp in components):
        return True
    return False


def normalize_parity_bracket_term(components):
    """
    Нормализует слагаемое скобки четности.

    Нормализация включает приведение каждой компоненты к канонической
    форме и сортировку компонент внутри слагаемого.

    Параметры:
    ----------
    components : list
        Список компонент связности

    Возвращает:
    -----------
    list
        Нормализованное слагаемое скобки четности
    """
    if not components:
        return [[]]

    normalized_components = []
    for component in components:
        if component:
            canonical_component = normalize_single_component(component)
            normalized_components.append(canonical_component)
        else:
            normalized_components.append([])

    normalized_components.sort(key=lambda x: (len(x), x))
    return normalized_components


def eliminate_duplicates_modulo_two(terms):
    """
    Устраняет дубликаты слагаемых по модулю 2.

    Параметры:
    ----------
    terms : list
        Список слагаемых скобки четности

    Возвращает:
    -----------
    list
        Список уникальных слагаемых по модулю 2
    """
    term_counter = Counter()

    for term in terms:
        term_signature = tuple(tuple(component) for component in term)
        term_counter[term_signature] += 1

    distinct_terms = []
    for term_signature, multiplicity in term_counter.items():
        if multiplicity % 2 == 1:
            term_components = [list(component) for component in term_signature]
            distinct_terms.append(term_components)

    return distinct_terms

# ============================================================================
# СКОБКА ЧЁТНОСТИ - ОСНОВНЫЕ ФУНКЦИИ
# ============================================================================

def compute_parity_bracket(diagram, verbose=False):
    """
    Вычисляет скобку четности для хордовой диаграммы.

    Скобка четности представляет собой формальную сумму по всем возможным
    разрешениям четных хорд, где каждое слагаемое соответствует классу
    изотопии полученного вырожденного узла.

    Параметры:
    ----------
    diagram : list
        Хордовая диаграмма свободного узла
    verbose : bool
        Если True, выводит подробные логи вычислений

    Возвращает:
    -----------
    list
        Список слагаемых скобки четности, где каждое слагаемое представлено
        списком компонент связности
    """
    # Идентификация четных хорд
    even_chords = identify_even_chords(diagram)
    if verbose:
        print(f"Четные хорды: {even_chords}")

    # Случай отсутствия четных хорд
    if not even_chords:
        normalized_diagram = normalize_chord_diagram(diagram)
        return [simplify_by_reidemeister_two([normalized_diagram])] if normalized_diagram else [[[]]]

    # Генерация всех возможных схем разрешений
    num_even_chords = len(even_chords)
    resolution_schemes = list(itertools.product(['break', 'flip'], repeat=num_even_chords))
    if verbose:
        print(f"Количество схем разрешений: {len(resolution_schemes)}")

    all_bracket_terms = []

    for scheme_index, resolution_scheme in enumerate(resolution_schemes):
        if verbose:
            print(f"\n--- Схема разрешений {scheme_index + 1}: {resolution_scheme} ---")

        # Построение технической диаграммы
        resolution_diagram = construct_resolution_diagram(diagram, even_chords, resolution_scheme)
        if verbose:
            print(f"Техническая диаграмма: {resolution_diagram}")

        # Извлечение компонент связности
        connected_components = extract_connected_components(resolution_diagram, even_chords, resolution_scheme, verbose)
        if verbose:
            print(f"Компоненты до упрощения: {connected_components}")

        # Упрощение вторым движением Рейдемейстера
        simplified_components = simplify_by_reidemeister_two(connected_components, verbose)
        if verbose:
            print(f"Компоненты после упрощения: {simplified_components}")

        # Проверка на нулевые слагаемые
        if is_null_term(simplified_components):
            if verbose:
                print("Пропущено (нулевое слагаемое)")
            continue

        # Нормализация и добавление слагаемого
        normalized_term = normalize_parity_bracket_term(simplified_components)
        all_bracket_terms.append(normalized_term)
        if verbose:
            print(f"Добавлено слагаемое: {normalized_term}")

    # Устранение дубликатов по модулю 2
    unique_terms = eliminate_duplicates_modulo_two(all_bracket_terms)

    if verbose:
        print(f"\n=== ФИНАЛЬНАЯ СКОБКА ЧЕТНОСТИ ===")
        print(f"Результат: {unique_terms}")

    return unique_terms


def compare_parity_brackets(first_bracket, second_bracket):
    """
    Сравнивает две скобки четности на равенство.

    Параметры:
    ----------
    first_bracket : list
        Первая скобка четности
    second_bracket : list
        Вторая скобка четности

    Возвращает:
    -----------
    bool
        True если скобки равны, иначе False
    """
    normalized_first = [normalize_bracket_term(term) for term in first_bracket]
    normalized_second = [normalize_bracket_term(term) for term in second_bracket]

    return find_bracket_isomorphism(normalized_first, normalized_second)


def normalize_bracket_term(term):
    """
    Нормализует слагаемое скобки четности для сравнения.

    Параметры:
    ----------
    term : list
        Слагаемое скобки четности

    Возвращает:
    -----------
    list
        Нормализованное слагаемое
    """
    normalized_components = []
    for component in term:
        if component:
            best_representation = find_canonical_cyclic_form(component)
            normalized_components.append(best_representation)
        else:
            normalized_components.append([])

    normalized_components.sort(key=lambda x: (len(x), x))
    return normalized_components


def find_canonical_cyclic_form(diagram):
    """
    Находит каноническую циклическую форму диаграммы.

    Параметры:
    ----------
    diagram : list
        Хордовая диаграмма

    Возвращает:
    -----------
    list
        Каноническое представление диаграммы
    """
    if not diagram:
        return []

    all_representations = []
    diagram_length = len(diagram)

    for shift_amount in range(diagram_length):
        shifted = diagram[shift_amount:] + diagram[:shift_amount]
        all_representations.append(tuple(shifted))

    mirror_image = diagram[::-1]
    for shift_amount in range(diagram_length):
        shifted_mirror = mirror_image[shift_amount:] + mirror_image[:shift_amount]
        all_representations.append(tuple(shifted_mirror))

    canonical_form = min(all_representations)
    return list(canonical_form)


def find_bracket_isomorphism(terms_a, terms_b):
    """
    Находит изоморфизм между двумя наборами слагаемых скобки четности.

    Параметры:
    ----------
    terms_a : list
        Первый набор слагаемых
    terms_b : list
        Второй набор слагаемых

    Возвращает:
    -----------
    bool
        True если наборы изоморфны, иначе False
    """
    if len(terms_a) != len(terms_b):
        return False

    for permutation in itertools.permutations(range(len(terms_b))):
        isomorphism_exists = True
        chord_correspondence = {}

        for idx_a, idx_b in enumerate(permutation):
            if not check_term_isomorphism(terms_a[idx_a], terms_b[idx_b], chord_correspondence):
                isomorphism_exists = False
                break

        if isomorphism_exists:
            return True

    return False


def check_term_isomorphism(term_a, term_b, existing_correspondence=None):
    """
    Проверяет изоморфность двух слагаемых скобки четности.

    Параметры:
    ----------
    term_a : list
        Первое слагаемое
    term_b : list
        Второе слагаемое
    existing_correspondence : dict, optional
        Существующее частичное соответствие между хордами

    Возвращает:
    -----------
    bool
        True если слагаемые изоморфны, иначе False
    """
    if len(term_a) != len(term_b):
        return False

    if existing_correspondence is None:
        existing_correspondence = {}

    chords_a = set(chord for component in term_a for chord in component)
    chords_b = set(chord for component in term_b for chord in component)

    unmatched_a = chords_a - set(existing_correspondence.keys())
    unmatched_b = chords_b - set(existing_correspondence.values())

    if len(unmatched_a) != len(unmatched_b):
        return False

    unmatched_a_sorted = sorted(unmatched_a)
    unmatched_b_sorted = sorted(unmatched_b)

    for correspondence_attempt in itertools.permutations(unmatched_b_sorted):
        test_correspondence = existing_correspondence.copy()
        for chord_idx, chord_a in enumerate(unmatched_a_sorted):
            test_correspondence[chord_a] = correspondence_attempt[chord_idx]

        correspondence_valid = True
        for comp_a, comp_b in zip(term_a, term_b):
            mapped_component_a = [test_correspondence[chord] for chord in comp_a]
            if find_canonical_cyclic_form(mapped_component_a) != find_canonical_cyclic_form(comp_b):
                correspondence_valid = False
                break

        if correspondence_valid:
            return True

    return False


def parity_analysis(candidates):
    """
    Анализ скобок четности для списка кандидатов.

    Вычисляет скобки четности для каждого кандидата и группирует их
    по эквивалентности.

    Параметры:
    ----------
    candidates : list of list
        Список диаграмм-кандидатов

    Возвращает:
    -----------
    tuple (list, list)
        classes: список классов эквивалентности (списки индексов узлов)
        brackets: список вычисленных скобок четности
    """
    # Вычисляем скобки четности
    brackets = []
    for i, diagram in enumerate(candidates):
        print(f"Вычисление для узла {i+1}: {diagram}")
        bracket = compute_parity_bracket(diagram, verbose=False)
        brackets.append(bracket)

    # Группируем
    classes = []
    used = set()

    for i in range(len(brackets)):
        if i in used:
            continue

        current_class = [i + 1]
        used.add(i)

        for j in range(i + 1, len(brackets)):
            if j in used:
                continue
            if compare_parity_brackets(brackets[i], brackets[j]):
                current_class.append(j + 1)
                used.add(j)

        classes.append(current_class)

    # Результаты
    print(f"\n{'='*60}")
    print("ГРУППИРОВКА ПО СКОБКАМ ЧЕТНОСТИ")
    print(f"{'='*60}\n")

    for class_idx, class_members in enumerate(classes, 1):
        print(f"Класс {class_idx}: узлы {class_members}")
        rep_idx = class_members[0] - 1
        print(f"  Скобка: {brackets[rep_idx]}")

    print(f"\nСтатистика: {len(classes)} классов из {len(candidates)} узлов")

    return classes, brackets

# ============================================================================
# z-ИНВАРИАНТ ГИБСОНА
# ============================================================================

def split_by_letter(diagram, letter):
    """
    Дробит Гауссово слово по заданной букве с учетом циклической природы.

    Параметры:
    ----------
    diagram : list
        Гауссово слово в виде списка меток хорд
    letter : int
        Буква (метка хорды), по которой производится разбиение

    Возвращает:
    -----------
    list of list
        Список из двух компонент: [[внутренние_элементы], [внешние_элементы]]
        где внутренние_элементы - между двумя вхождениями буквы,
        внешние_элементы - всё остальное в циклическом порядке

    Примеры:
    --------
    >>> split_by_letter([1, 2, 3, 1, 2, 3], 1)
    [[2, 3], [2, 3]]
    >>> split_by_letter([1, 2, 1, 3, 2, 3], 1)
    [[2], [3, 2, 3]]
    """
    positions = get_indices(diagram, letter)

    if len(positions) != 2:
        raise ValueError(f"Буква '{letter}' должна встречаться ровно 2 раза в диаграмме")

    pos1, pos2 = sorted(positions)

    # Внутренние элементы (между позициями)
    inner_elements = diagram[pos1 + 1:pos2]

    # Внешние элементы (циклически после pos2 и до pos1)
    if pos2 + 1 < len(diagram):
        outer_elements = diagram[pos2 + 1:] + diagram[:pos1]
    else:
        outer_elements = diagram[:pos1]

    return [inner_elements, outer_elements]


def split_by_all_letters(diagram):
    """
    Дробит Гауссово слово по каждой уникальной букве.

    Параметры:
    ----------
    diagram : list
        Гауссово слово в виде списка меток хорд

    Возвращает:
    -----------
    list of list of list
        Список раздробленных слов для каждой уникальной буквы.
        Каждый элемент - результат split_by_letter для соответствующей буквы.

    Примеры:
    --------
    >>> split_by_all_letters([1, 2, 3, 1, 2, 3])
    [
        [[2, 3], [2, 3]],
        [[3, 1], [3, 1]],
        [[1, 2], [1, 2]]
    ]
    """
    if not diagram:
        return []

    unique_letters = sorted(set(diagram))

    results = []
    for letter in unique_letters:
        split_result = split_by_letter(diagram, letter)
        results.append(split_result)

    return results


def identify_even_chords_in_phrase(phrase):
    """
    Определяет четные хорды в гауссовой фразе [[C1], [C2]].

    Четная хорда = хорда, оба конца которой находятся в одной компоненте
    (либо в C1, либо в C2).

    Параметры:
    ----------
    phrase : list of list
        Гауссова фраза [[C1], [C2]]

    Возвращает:
    -----------
    list
        Список меток четных хорд
    """
    if not phrase:
        raise ValueError("Фраза не может быть пустой")

    if len(phrase) != 2:
        raise ValueError(f"Фраза должна содержать 2 компоненты, получено {len(phrase)}")

    C1, C2 = phrase[0], phrase[1]

    if not isinstance(C1, list) or not isinstance(C2, list):
        raise ValueError("Компоненты фразы должны быть списками")

    all_chords = set(C1 + C2)
    even_chords = []

    for chord in all_chords:
        count_in_C1 = C1.count(chord)
        count_in_C2 = C2.count(chord)

        if count_in_C1 == 2 or count_in_C2 == 2:
            even_chords.append(chord)

    return sorted(even_chords)


def get_even_chords_for_component(component):
    """
    Находит хорды, которые встречаются в компоненте 2 раза
    (имеют оба конца в этой компоненте).

    Параметры:
    ----------
    component : list
        Компонента гауссовой фразы (C1 или C2)

    Возвращает:
    -----------
    list
        Отсортированный список четных хорд в компоненте
    """
    if not component:
        return []

    even_chords = []
    for chord in set(component):
        if component.count(chord) == 2:
            even_chords.append(chord)

    return sorted(even_chords)


def generate_resolution_schemes_for_phrase(phrase):
    """
    Генерирует все схемы разрешений для гауссовой фразы [[C1], [C2]].

    Параметры:
    ----------
    phrase : list of list
        Гауссова фраза [[C1], [C2]]

    Возвращает:
    -----------
    list of tuple
        Список пар (scheme_C1, scheme_C2), где каждая схема -
        кортеж из 'break' и 'flip'
    """
    if len(phrase) != 2:
        raise ValueError(f"Фраза должна содержать 2 компоненты, получено {len(phrase)}")

    C1, C2 = phrase[0], phrase[1]

    even_C1 = get_even_chords_for_component(C1)
    even_C2 = get_even_chords_for_component(C2)

    schemes_C1 = list(itertools.product(['break', 'flip'], repeat=len(even_C1)))
    schemes_C2 = list(itertools.product(['break', 'flip'], repeat=len(even_C2)))

    all_schemes = list(itertools.product(schemes_C1, schemes_C2))

    return all_schemes


def construct_resolution_diagram_for_component(component, even_chords, resolution_scheme):
    """
    Строит техническую диаграмму для компоненты фразы.
    Адаптированная версия оригинальной construct_resolution_diagram.

    Параметры:
    ----------
    component : list
        Компонента фразы (C1 или C2)
    even_chords : list
        Четные хорды в этой компоненте (встречаются 2 раза в компоненте)
    resolution_scheme : tuple
        Схема разрешений для четных хорд ('break'/'flip')

    Возвращает:
    -----------
    list
        Техническая диаграмма компоненты
    """
    if len(even_chords) != len(resolution_scheme):
        raise ValueError(f"Количество четных хорд ({len(even_chords)}) не совпадает "
                         f"с длиной схемы разрешений ({len(resolution_scheme)})")

    resolution_diagram = []
    resolution_mapping = dict(zip(even_chords, resolution_scheme))

    for chord in component:
        if chord in even_chords:
            resolution_type = resolution_mapping[chord]

            if resolution_type == 'break':
                appearances_count = sum(1 for x in resolution_diagram if abs(x) == chord)
                if appearances_count == 0:
                    resolution_diagram.extend([chord, -chord])
                elif appearances_count == 2:
                    resolution_diagram.extend([-chord, chord])
            else:  # resolution_type == 'flip'
                appearances_count = sum(1 for x in resolution_diagram if abs(x) == chord)
                if appearances_count == 0:
                    resolution_diagram.extend([chord, -chord])
                elif appearances_count == 2:
                    resolution_diagram.extend([chord, -chord])
        else:
            resolution_diagram.append(chord)

    return resolution_diagram


def compute_components_for_phrase_scheme(phrase, scheme_pair, verbose=False):
    """
    Вычисляет компоненты связности для гауссовой фразы с заданной схемой разрешений.

    Параметры:
    ----------
    phrase : list of list
        Гауссова фраза [[C1], [C2]]
    scheme_pair : tuple
        Пара схем разрешений (scheme_C1, scheme_C2)
    verbose : bool
        Если True, выводит подробные логи

    Возвращает:
    -----------
    list
        Объединенный список компонент связности из C1 и C2
        (без упрощения вторым движением и нормализации)
    """
    if len(phrase) != 2:
        raise ValueError(f"Фраза должна содержать 2 компоненты, получено {len(phrase)}")

    if len(scheme_pair) != 2:
        raise ValueError(f"Пара схем должна содержать 2 схемы, получено {len(scheme_pair)}")

    C1, C2 = phrase[0], phrase[1]
    scheme_C1, scheme_C2 = scheme_pair

    even_C1 = get_even_chords_for_component(C1)
    even_C2 = get_even_chords_for_component(C2)

    if verbose:
        print(f"compute_components_for_phrase_scheme:")
        print(f"  C1: {C1}")
        print(f"  C2: {C2}")
        print(f"  Четные в C1: {even_C1}, схема: {scheme_C1}")
        print(f"  Четные в C2: {even_C2}, схема: {scheme_C2}")

    if len(even_C1) != len(scheme_C1):
        raise ValueError(f"Для C1: {len(even_C1)} четных хорд, но {len(scheme_C1)} схем")
    if len(even_C2) != len(scheme_C2):
        raise ValueError(f"Для C2: {len(even_C2)} четных хорд, но {len(scheme_C2)} схем")

    tech_C1 = construct_resolution_diagram_for_component(C1, even_C1, scheme_C1)
    tech_C2 = construct_resolution_diagram_for_component(C2, even_C2, scheme_C2)

    if verbose:
        print(f"  Тех. диаграмма C1: {tech_C1}")
        print(f"  Тех. диаграмма C2: {tech_C2}")

    components_C1 = extract_connected_components(tech_C1, even_C1, scheme_C1, verbose)
    components_C2 = extract_connected_components(tech_C2, even_C2, scheme_C2, verbose)

    if verbose:
        print(f"  Компоненты из C1: {components_C1}")
        print(f"  Компоненты из C2: {components_C2}")

    all_components = components_C1 + components_C2

    if verbose:
        print(f"  Все компоненты (объединенные): {all_components}")

    return all_components


def compute_parity_bracket_for_phrase(phrase, verbose=False):
    """
    Вычисляет скобку четности для гауссовой фразы [[C1], [C2]].

    Параметры:
    ----------
    phrase : list of list
        Гауссова фраза [[C1], [C2]]
    verbose : bool
        Если True, выводит подробные логи

    Возвращает:
    -----------
    list
        Список слагаемых скобки четности (после устранения дубликатов по модулю 2)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Вычисление скобки четности для фразы: {phrase}")
        print(f"{'='*60}")

    all_schemes = generate_resolution_schemes_for_phrase(phrase)

    if verbose:
        print(f"Всего схем разрешений: {len(all_schemes)}")
        if len(all_schemes) <= 10:
            print(f"Схемы: {all_schemes}")

    all_bracket_terms = []

    for i, scheme_pair in enumerate(all_schemes):
        if verbose:
            print(f"\n--- Схема {i+1}/{len(all_schemes)}: {scheme_pair} ---")

        try:
            components = compute_components_for_phrase_scheme(phrase, scheme_pair, verbose)

            if verbose:
                print(f"  Компоненты после извлечения: {components}")

            simplified = simplify_by_reidemeister_two(components, verbose)

            if verbose:
                print(f"  После упрощения вторым движением: {simplified}")

            if is_null_term(simplified):
                if verbose:
                    print(f"  ⚠ Нулевое слагаемое, пропускаем")
                continue

            normalized = normalize_parity_bracket_term(simplified)

            if verbose:
                print(f"  Нормализованное слагаемое: {normalized}")

            all_bracket_terms.append(normalized)

        except Exception as e:
            if verbose:
                print(f"  ⚠ Ошибка при обработке схемы {scheme_pair}: {e}")
            continue

    if verbose:
        print(f"\n{'='*60}")
        print(f"Результат до устранения дубликатов:")
        print(f"Количество слагаемых: {len(all_bracket_terms)}")
        for i, term in enumerate(all_bracket_terms):
            print(f"  Слагаемое {i+1}: {term}")
        print(f"{'='*60}")

    unique_terms = eliminate_duplicates_modulo_two(all_bracket_terms)

    if verbose:
        print(f"\n{'='*60}")
        print(f"ФИНАЛЬНАЯ СКОБКА ЧЕТНОСТИ:")
        print(f"Слагаемых после устранения дубликатов: {len(unique_terms)}")
        if unique_terms:
            for i, term in enumerate(unique_terms):
                print(f"  Слагаемое {i+1}: {term}")
        else:
            print(f"  Пустая скобка (только нулевые слагаемые)")
        print(f"{'='*60}")

    return unique_terms

def gibson_invariant_terms(diagram):
    """
    Вычисляет и выводит инвариант Гибсона для диаграммы.
    
    Для каждой буквы (хорды) вычисляется скобка четности соответствующей фразы.
    Выводятся только слагаемые инварианта Гибсона.

    Корректен для диаграм с четным кол-вом хорд.
    
    Параметры:
    ----------
    diagram : list of int
        Хордовая диаграмма
    """
    print("Слагаемые инварианта Гибсона:")
    
    letter_index = 1
    
    for letter in sorted(set(diagram)):
        # Разбиваем диаграмму по букве
        phrase = split_by_letter(diagram, letter)
        
        # Вычисляем скобку четности для фразы
        bracket = compute_parity_bracket_for_phrase(phrase, verbose=False)
        
        # Выводим в нужном формате
        if bracket:
            # Форматируем слагаемые
            terms_str = " + ".join([str(term) for term in bracket])
            print(f"{letter_index}: {terms_str}")
        else:
            print(f"{letter_index}: ∅")
        
        letter_index += 1

# ============================================================================
# ВИЗУАЛИЗАЦИЯ С ХОРДАМИ (with_chords версии)
# ============================================================================

def filter_results_with_chords(results_with_chords):
    """
    Фильтрует результаты с сохранением хорд, оставляя только уникальные диаграммы.

    Параметры:
    ----------
    results_with_chords : list of tuple
        Список кортежей (диаграмма, хорды)

    Возвращает:
    -----------
    list of tuple
        Список уникальных результатов
    """
    if not results_with_chords:
        return []

    unique_results = []
    seen_canonical = set()

    for diagram, chords in results_with_chords:
        canonical = _compute_canonical_form(diagram)
        canonical_key = tuple(tuple(indices) for indices in canonical)

        if canonical_key not in seen_canonical:
            seen_canonical.add(canonical_key)
            unique_results.append((diagram, chords))

    return unique_results


def mod_1_with_chords(diagram):
    """
    Применяет первое уменьшающее движение Рейдемейстера (Ω1) с сохранением информации о хордах.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of tuple
        Список кортежей (модифицированная_диаграмма, участвующие_хорды)
    """
    results = [(diagram.copy(), None)]

    for position in range(len(diagram)):
        current_chord = diagram[position]
        previous_chord = diagram[position - 1]

        if current_chord == previous_chord:
            modified_diagram = diagram.copy()

            if position == 0:
                del modified_diagram[-1]
                del modified_diagram[0]
            else:
                del modified_diagram[position - 1]
                del modified_diagram[position - 1]

            involved_chords = [current_chord]
            results.append((modified_diagram, involved_chords))

    return results


def mod_2_reducing_with_chords(diagram):
    """
    Применяет второе уменьшающее движение Рейдемейстера (Ω2) с сохранением информации о хордах.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of tuple
        Список кортежей (модифицированная_диаграмма, участвующие_хорды)
    """
    results = [(diagram.copy(), None)]

    for i in range(len(diagram) - 2):
        chord_a = diagram[i]
        chord_b = diagram[i - 1]

        if i == 0:
            for j in range(i + 2, len(diagram) - 1):
                chord_c = diagram[j]
                chord_d = diagram[j - 1]

                if (chord_c == chord_a and chord_d == chord_b) or \
                   (chord_c == chord_b and chord_d == chord_a):

                    modified_diagram = diagram.copy()
                    positions_to_remove = sorted([i - 1, i, j - 1, j], reverse=True)
                    for pos in positions_to_remove:
                        del modified_diagram[pos]

                    involved_chords = sorted([chord_a, chord_b])
                    results.append((modified_diagram, involved_chords))
        else:
            for j in range(i + 2, len(diagram)):
                chord_c = diagram[j]
                chord_d = diagram[j - 1]

                if (chord_c == chord_a and chord_d == chord_b) or \
                   (chord_c == chord_b and chord_d == chord_a):

                    modified_diagram = diagram.copy()
                    positions_to_remove = sorted([i - 1, i, j - 1, j], reverse=True)
                    for pos in positions_to_remove:
                        del modified_diagram[pos]

                    involved_chords = sorted([chord_a, chord_b])
                    results.append((modified_diagram, involved_chords))

    return results


def mod_2_non_reducing_with_chords(diagram):
    """
    Применяет второе неуменьшающее движение Рейдемейстера (Ω2n) с сохранением информации о хордах.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of tuple
        Список кортежей (модифицированная_диаграмма, участвующие_хорды)
    """
    results = [(diagram.copy(), None)]

    for i in range(len(diagram) - 2):
        chord_a = diagram[i]
        chord_b = diagram[i - 1]

        if i == 0:
            for j in range(i + 2, len(diagram) - 1):
                chord_c = diagram[j]
                chord_d = diagram[j - 1]

                if (chord_c == chord_a and chord_d == chord_b) or \
                   (chord_c == chord_b and chord_d == chord_a):

                    modified_diagram = diagram.copy()
                    modified_diagram[i] = chord_b
                    modified_diagram[i - 1] = chord_a

                    involved_chords = sorted([chord_a, chord_b])
                    results.append((modified_diagram, involved_chords))
        else:
            for j in range(i + 2, len(diagram)):
                chord_c = diagram[j]
                chord_d = diagram[j - 1]

                if (chord_c == chord_a and chord_d == chord_b) or \
                   (chord_c == chord_b and chord_d == chord_a):

                    modified_diagram = diagram.copy()
                    modified_diagram[i] = chord_b
                    modified_diagram[i - 1] = chord_a

                    involved_chords = sorted([chord_a, chord_b])
                    results.append((modified_diagram, involved_chords))

    return results


def mod_3_with_chords(diagram):
    """
    Применяет третье движение Рейдемейстера (Ω3) с сохранением информации о хордах.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of tuple
        Список кортежей (модифицированная_диаграмма, участвующие_хорды)
    """
    results = [(diagram.copy(), None)]
    working_diagram = diagram.copy()
    diagram_length = len(working_diagram)

    for shift_count in range(diagram_length):
        chord_indices = _get_sorted_chord_indices(working_diagram)

        if chord_indices:
            base_chord = chord_indices[0]
            x1, x2 = base_chord

            _check_type_1_to_2_with_chords(working_diagram, x1, x2, chord_indices, results)
            _check_type_2_to_1_with_chords(working_diagram, x1, x2, chord_indices, results)
            _check_type_3_to_4_with_chords(working_diagram, x1, x2, chord_indices, results)
            _check_type_4_to_3_with_chords(working_diagram, x1, x2, chord_indices, results)

        working_diagram = shift(working_diagram, 1)

    return results


def _check_type_1_to_2_with_chords(diagram, x1, x2, chord_indices, results):
    """Проверяет третье движение типа 1->2 с сохранением хорд."""
    n = len(diagram)

    for position in range(n):
        condition_1 = [x1 + 1, position - 1] in chord_indices
        condition_2 = [x2 - 1, position] in chord_indices
        condition_3 = ([x1, x2] != [x1 + 1, position - 1])
        condition_4 = ([x1, x2] != [x2 - 1, position])
        condition_5 = ([x1 + 1, position - 1] != [x2 - 1, position])

        if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
            new_diagram = diagram.copy()
            new_diagram[x1] = diagram[x1 + 1]
            new_diagram[x1 + 1] = diagram[x1]
            new_diagram[x2 - 1] = diagram[x2]
            new_diagram[x2] = diagram[x2 - 1]
            new_diagram[position] = diagram[position - 1]
            new_diagram[position - 1] = diagram[position]

            chord_A = diagram[x1]
            chord_B = diagram[x1 + 1]
            chord_C = diagram[x2 - 1]

            involved_chords = sorted([chord_A, chord_B, chord_C])
            results.append((new_diagram, involved_chords))


def _check_type_2_to_1_with_chords(diagram, x1, x2, chord_indices, results):
    """Проверяет третье движение типа 2->1 с сохранением хорд."""
    n = len(diagram)

    for position in range(n):
        condition_1 = [x1 + 1, position - 1] in chord_indices
        condition_2 = [position, x2 - 1] in chord_indices
        condition_3 = ([x1, x2] != [x1 + 1, position - 1])
        condition_4 = ([x1, x2] != [position, x2 - 1])
        condition_5 = ([x1 + 1, position - 1] != [position, x2 - 1])

        if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
            new_diagram = diagram.copy()
            new_diagram[x1] = diagram[x1 + 1]
            new_diagram[x1 + 1] = diagram[x1]
            new_diagram[x2 - 1] = diagram[x2]
            new_diagram[x2] = diagram[x2 - 1]
            new_diagram[position] = diagram[position - 1]
            new_diagram[position - 1] = diagram[position]

            chord_A = diagram[x1]
            chord_B = diagram[x1 + 1]
            chord_C = diagram[position]

            involved_chords = sorted([chord_A, chord_B, chord_C])
            results.append((new_diagram, involved_chords))


def _check_type_3_to_4_with_chords(diagram, x1, x2, chord_indices, results):
    """Проверяет третье движение типа 3->4 с сохранением хорд."""
    n = len(diagram)

    for position in range(n):
        condition_1 = [x1 + 1, position - 1] in chord_indices
        condition_2 = [x2 + 1, position] in chord_indices
        condition_3 = ([x1, x2] != [x1 + 1, position - 1])
        condition_4 = ([x1, x2] != [x2 + 1, position])
        condition_5 = ([x1 + 1, position - 1] != [x2 + 1, position])

        if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
            new_diagram = diagram.copy()
            new_diagram[x1] = diagram[x1 + 1]
            new_diagram[x1 + 1] = diagram[x1]
            new_diagram[x2 + 1] = diagram[x2]
            new_diagram[x2] = diagram[x2 + 1]
            new_diagram[position] = diagram[position - 1]
            new_diagram[position - 1] = diagram[position]

            chord_A = diagram[x1]
            chord_B = diagram[x1 + 1]
            chord_C = diagram[x2 + 1]

            involved_chords = sorted([chord_A, chord_B, chord_C])
            results.append((new_diagram, involved_chords))


def _check_type_4_to_3_with_chords(diagram, x1, x2, chord_indices, results):
    """Проверяет третье движение типа 4->3 с сохранением хорд."""
    n = len(diagram)

    for position in range(n):
        condition_1 = [x1 + 1, position + 1] in chord_indices
        condition_2 = [position, x2 - 1] in chord_indices
        condition_3 = ([x1, x2] != [x1 + 1, position + 1])
        condition_4 = ([x1, x2] != [position, x2 - 1])
        condition_5 = ([x1 + 1, position + 1] != [position, x2 - 1])

        if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
            new_diagram = diagram.copy()
            new_diagram[x1] = diagram[x1 + 1]
            new_diagram[x1 + 1] = diagram[x1]
            new_diagram[x2 - 1] = diagram[x2]
            new_diagram[x2] = diagram[x2 - 1]
            new_diagram[position] = diagram[position + 1]
            new_diagram[position + 1] = diagram[position]

            chord_A = diagram[x1]
            chord_B = diagram[x1 + 1]
            chord_C = diagram[position]

            involved_chords = sorted([chord_A, chord_B, chord_C])
            results.append((new_diagram, involved_chords))


def mod_1_filtered(diagram):
    """
    Применяет Ω1 с фильтрацией уникальных результатов.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of tuple
        Список уникальных результатов после фильтрации
    """
    results = mod_1_with_chords(diagram)
    return filter_results_with_chords(results)


def mod_2_reducing_filtered(diagram):
    """
    Применяет Ω2 уменьшающее с фильтрацией уникальных результатов.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of tuple
        Список уникальных результатов после фильтрации
    """
    results = mod_2_reducing_with_chords(diagram)
    return filter_results_with_chords(results)


def mod_2_non_reducing_filtered(diagram):
    """
    Применяет Ω2 неуменьшающее с фильтрацией уникальных результатов.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of tuple
        Список уникальных результатов после фильтрации
    """
    results = mod_2_non_reducing_with_chords(diagram)
    return filter_results_with_chords(results)


def mod_3_filtered(diagram):
    """
    Применяет Ω3 с фильтрацией уникальных результатов.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма

    Возвращает:
    -----------
    list of tuple
        Список уникальных результатов после фильтрации
    """
    results = mod_3_with_chords(diagram)
    return filter_results_with_chords(results)


def trace_simplification_with_chords(diagram, timeout=30):
    """
    Трассирует процесс упрощения диаграммы с сохранением информации о хордах.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма
    timeout : int
        Максимальное время выполнения в секундах

    Возвращает:
    -----------
    list of tuple
        Список кортежей (диаграмма, тип_движения, участвующие_хорды)
        или None если путь не найден
    """
    start_time = time.time()

    queue = deque()
    visited = set()
    parent = {}
    move_info = {}

    initial_tuple = tuple(diagram)
    queue.append((diagram, 0, "start", []))
    visited.add(initial_tuple)
    parent[initial_tuple] = None
    move_info[initial_tuple] = ("start", [])

    while queue and (time.time() - start_time) < timeout:
        current_diag, depth, move_type, chords = queue.popleft()
        current_tuple = tuple(current_diag)

        if len(current_diag) == 0:
            break

        if depth > 40:
            continue

        # Ω1 движения
        omega1_results = mod_1_filtered(current_diag)
        for new_diag, involved_chords in omega1_results:
            if involved_chords is None:
                continue
            new_tuple = tuple(new_diag)
            if new_tuple not in visited:
                visited.add(new_tuple)
                parent[new_tuple] = current_diag
                move_info[new_tuple] = ("Ω1", involved_chords)
                queue.append((new_diag, depth + 1, "Ω1", involved_chords))

        # Ω2 уменьшающие движения
        omega2_reducing_results = mod_2_reducing_filtered(current_diag)
        for new_diag, involved_chords in omega2_reducing_results:
            if involved_chords is None:
                continue
            new_tuple = tuple(new_diag)
            if new_tuple not in visited:
                visited.add(new_tuple)
                parent[new_tuple] = current_diag
                move_info[new_tuple] = ("Ω2", involved_chords)
                queue.append((new_diag, depth + 1, "Ω2", involved_chords))

        # Ω2 неуменьшающие и Ω3 (только на небольших глубинах)
        if depth < 6:
            # Ω2 неуменьшающие
            omega2_non_results = mod_2_non_reducing_filtered(current_diag)
            for new_diag, involved_chords in omega2_non_results[:3]:
                if involved_chords is None:
                    continue
                new_tuple = tuple(new_diag)
                if new_tuple not in visited:
                    visited.add(new_tuple)
                    parent[new_tuple] = current_diag
                    move_info[new_tuple] = ("Ω2n", involved_chords)
                    queue.append((new_diag, depth + 1, "Ω2n", involved_chords))

            # Ω3 движения
            omega3_results = mod_3_filtered(current_diag)
            for new_diag, involved_chords in omega3_results[:3]:
                if involved_chords is None:
                    continue
                new_tuple = tuple(new_diag)
                if new_tuple not in visited:
                    visited.add(new_tuple)
                    parent[new_tuple] = current_diag
                    move_info[new_tuple] = ("Ω3", involved_chords)
                    queue.append((new_diag, depth + 1, "Ω3", involved_chords))

    # Находим наименьшую достигнутую диаграмму
    if not visited:
        return None

    best_tuple = min(visited, key=lambda x: len(x))

    # Восстанавливаем путь
    path = []
    current = best_tuple

    while current is not None:
        current_diag = list(current)
        if current in move_info:
            move_type, chords = move_info[current]
            path.append((current_diag, move_type, chords))
        else:
            path.append((current_diag, "start", []))

        if current in parent:
            current = tuple(parent[current]) if parent[current] is not None else None
        else:
            current = None

    return list(reversed(path))

# ============================================================================
# РИСОВАНИЕ ХОРДОВЫХ ДИАГРАММ
# ============================================================================

def draw_chord_diagram(diagram, ax=None, size=120, highlight_chords=None):
    """
    Рисует хордовую диаграмму с возможностью выделения хорд.

    Параметры:
    ----------
    diagram : list of int
        Хордовая диаграмма
    ax : matplotlib.axes.Axes, optional
        Ось для рисования (если None, создается новая)
    size : int
        Размер диаграммы в пикселях
    highlight_chords : list, optional
        Список хорд для выделения красным цветом

    Возвращает:
    -----------
    matplotlib.axes.Axes
        Ось с нарисованной диаграммой
    """
    if ax is None:
        dpi = 100
        figsize = size / dpi
        fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
        ax.set_aspect('equal')

    if not diagram:
        circle = Circle((0, 0), 0.9, fill=False, color='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axis('off')
        return ax

    radius = 0.9
    circle = Circle((0, 0), radius, fill=False, color='black', linewidth=1.5)
    ax.add_patch(circle)

    n_points = len(diagram)
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x_points = radius * np.cos(angles)
    y_points = radius * np.sin(angles)

    chord_positions = {}
    for i, chord in enumerate(diagram):
        chord_positions.setdefault(chord, []).append(i)

    if highlight_chords is None:
        highlight_chords = []

    # Рисуем хорды
    for chord, positions in chord_positions.items():
        if len(positions) == 2:
            i1, i2 = positions
            x1, y1 = x_points[i1], y_points[i1]
            x2, y2 = x_points[i2], y_points[i2]

            if chord in highlight_chords:
                color, width = 'red', 2.5
            else:
                color, width = 'black', 1.2

            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.8)

    # Рисуем точки и подписи
    for i, (x, y, chord) in enumerate(zip(x_points, y_points, diagram)):
        if chord in highlight_chords:
            point_color, edge_color, point_size = 'red', 'darkred', 6
            text_color = 'red'
        else:
            point_color, edge_color, point_size = 'black', 'black', 4
            text_color = 'black'

        ax.plot(x, y, 'o', color=point_color, markersize=point_size,
                markeredgecolor=edge_color, markeredgewidth=1.0, zorder=3)

        angle = np.arctan2(y, x)
        label_distance = 1.25
        label_x = x * label_distance
        label_y = y * label_distance

        offset_x = np.sin(angle) * 0.05
        offset_y = -np.cos(angle) * 0.05

        ax.text(label_x + offset_x, label_y + offset_y, str(chord),
                fontsize=6, color=text_color, fontweight='bold',
                ha='center', va='center', zorder=4)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')

    return ax


def _draw_diagram_with_fixed_aspect(diagram, ax, size, highlight_chords=None):
    """
    Рисует диаграмму с фиксированным соотношением сторон для встраивания.

    Параметры:
    ----------
    diagram : list of int
        Хордовая диаграмма
    ax : matplotlib.axes.Axes
        Ось для рисования
    size : int
        Размер диаграммы
    highlight_chords : list, optional
        Список хорд для выделения
    """
    draw_chord_diagram(diagram, ax=ax, size=size, highlight_chords=highlight_chords)


def visualize_simplification(simplification_path, max_cols=8, diagram_size=120):
    """
    Визуализирует последовательность упрощения хордовой диаграммы.

    Параметры:
    ----------
    simplification_path : list of tuple
        Список кортежей (диаграмма, тип_движения, участвующие_хорды)
    max_cols : int
        Максимальное количество колонок
    diagram_size : int
        Размер диаграмм в пикселях

    Возвращает:
    -----------
    matplotlib.figure.Figure
        Фигура с визуализацией
    """
    if not simplification_path:
        return None

    n_steps = len(simplification_path)
    cols = min(max_cols, n_steps)
    rows = (n_steps + cols - 1) // cols

    dpi = 100
    fig_width = cols * diagram_size / dpi
    fig_height = rows * diagram_size / dpi

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=dpi)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(rows):
        for j in range(cols):
            axes[i, j].set_aspect('equal')

    for idx, (diagram, move_type, involved_chords) in enumerate(simplification_path):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Если это НЕ последняя диаграмма, выделяем хорды,
        # которые будут изменены в СЛЕДУЮЩЕМ шаге
        if idx < len(simplification_path) - 1:
            _, next_move_type, next_involved_chords = simplification_path[idx + 1]
            highlight_chords = next_involved_chords
        else:
            highlight_chords = []

        draw_chord_diagram(diagram, ax=ax, size=diagram_size,
                          highlight_chords=highlight_chords)

        # Номер шага
        ax.text(0.95, 0.95, f"{idx + 1}", fontsize=8, fontweight='bold',
                ha='right', va='top', transform=ax.transAxes,
                bbox=dict(boxstyle="circle,pad=0.2",
                         facecolor='white', edgecolor='gray', alpha=0.7))

        # Символ движения
        if move_type != "start":
            symbols = {
                "Ω1": "Ω₁",
                "Ω2": "Ω₂",
                "Ω2n": "Ω₂ⁿ",
                "Ω3": "Ω₃"
            }
            symbol = symbols.get(move_type, "")
            if symbol:
                ax.text(0.05, 0.95, symbol, fontsize=7, fontweight='bold',
                        ha='left', va='top', transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.1",
                                 facecolor='yellow', alpha=0.7))

    total_cells = rows * cols
    for idx in range(n_steps, total_cells):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
    return fig


def show_simplification(diagram, timeout=15, max_cols=8):
    """
    Показывает процесс упрощения диаграммы без текстового вывода.

    Параметры:
    ----------
    diagram : list of int
        Исходная хордовая диаграмма
    timeout : int
        Время на поиск упрощения в секундах
    max_cols : int
        Максимальное количество колонок

    Возвращает:
    -----------
    matplotlib.figure.Figure or None
        Фигура с визуализацией или None
    """
    path = trace_simplification_with_chords(diagram, timeout=timeout)

    if not path:
        return None

    fig = visualize_simplification(path, max_cols=max_cols)
    plt.show()
    return fig


# ============================================================================
# СОХРАНЕНИЕ В PDF
# ============================================================================

def _draw_simplification_sequence(ax, diagram, path, diagram_size, diagram_number):
    """
    Рисует последовательность упрощения диаграммы.
    Если не помещается в одну строку - переносит на новую, сохраняя размер диаграмм.

    Параметры:
    ----------
    ax : matplotlib.axes.Axes
        Ось для рисования
    diagram : list
        Исходная диаграмма
    path : list
        Путь упрощения
    diagram_size : int
        Размер диаграммы
    diagram_number : int
        Номер диаграммы для подписи
    """
    ax.text(0.02, 0.95, f"Диаграмма {diagram_number}: {diagram}",
            fontsize=8, fontweight='bold', transform=ax.transAxes)

    n_steps = len(path)
    cell_width = 0.12
    cell_height = 0.7

    available_width = 0.96
    diagrams_per_row = int(available_width / cell_width)

    rows_needed = (n_steps + diagrams_per_row - 1) // diagrams_per_row

    row_height = 0.7
    row_spacing = 0.1

    start_y = 0.15

    for row in range(rows_needed):
        y_center = start_y - row * (row_height + row_spacing)

        start_idx = row * diagrams_per_row
        end_idx = min((row + 1) * diagrams_per_row, n_steps)

        for idx in range(start_idx, end_idx):
            col = idx - start_idx
            diag, move_type, chords = path[idx]
            x_center = 0.02 + col * cell_width

            subax = ax.inset_axes([x_center, y_center, cell_width * 0.9, row_height])
            subax.set_aspect('equal', adjustable='box')

            if idx < len(path) - 1:
                _, _, next_chords = path[idx + 1]
                highlight_chords = next_chords
            else:
                highlight_chords = chords

            _draw_diagram_with_fixed_aspect(diag, subax, diagram_size, highlight_chords)

            subax.text(0.95, 0.95, f"{idx+1}", fontsize=7, fontweight='bold',
                       ha='right', va='top', transform=subax.transAxes,
                       bbox=dict(boxstyle="circle,pad=0.15",
                                facecolor='white', edgecolor='gray', alpha=0.7))

            if move_type != "start":
                symbols = {"Ω1": "Ω₁", "Ω2": "Ω₂", "Ω2n": "Ω₂ⁿ", "Ω3": "Ω₃"}
                symbol = symbols.get(move_type, "")
                if symbol:
                    subax.text(0.05, 0.95, symbol, fontsize=6, fontweight='bold',
                               ha='left', va='top', transform=subax.transAxes,
                               bbox=dict(boxstyle="round,pad=0.1",
                                        facecolor='yellow', alpha=0.7))

    ax.axis('off')


def save_simplifications_to_pdf(diagrams_list, filename="simplifications.pdf",
                                diagram_size=105, timeout=30):
    """
    Сохраняет упрощения хордовых диаграмм в PDF файл.

    Параметры:
    ----------
    diagrams_list : list of list
        Список хордовых диаграмм для визуализации
    filename : str
        Имя выходного PDF файла
    diagram_size : int
        Размер каждой диаграммы в пикселях
    timeout : int
        Время на поиск упрощения для каждой диаграммы

    Возвращает:
    -----------
    dict
        Словарь с результатами обработки каждой диаграммы
    """
    results = {}

    with PdfPages(filename) as pdf:
        diagrams_per_page = 4

        for i in range(0, len(diagrams_list), diagrams_per_page):
            page_diagrams = diagrams_list[i:i + diagrams_per_page]

            fig = plt.figure(figsize=(11.69, 8.27))

            for j, diagram in enumerate(page_diagrams):
                idx = i + j
                print(f"Обработка диаграммы {idx+1}/{len(diagrams_list)}: {diagram}")

                path = trace_simplification_with_chords(diagram, timeout=timeout)

                if not path:
                    print(f"  Не удалось найти упрощение для {diagram}")
                    results[idx] = {'success': False, 'reason': 'Путь не найден'}

                    ax = plt.subplot2grid((diagrams_per_page, 1), (j, 0))
                    ax.text(0.02, 0.5, f"Диаграмма {idx+1}: {diagram}",
                            fontsize=8, fontweight='bold', transform=ax.transAxes)
                    ax.axis('off')
                    continue

                ax = plt.subplot2grid((diagrams_per_page, 1), (j, 0))

                _draw_simplification_sequence(ax, diagram, path, diagram_size, idx + 1)

                results[idx] = {
                    'success': True,
                    'steps': len(path),
                    'final_size': len(path[-1][0])
                }

            plt.tight_layout(h_pad=2.0)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    return results

