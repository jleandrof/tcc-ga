# Genetic Algorithm for the Traveling Salesman Problem (TSP) using Holland's Method

import random
from sys import maxsize as MAX_INT

DEFAULT_GENE_LENGTH = 5
DEFAULT_POPULATION_SIZE = 5
DEFAULT_GENERATION_COUNT = 200

MUTATION_RATE = 0.1

CITIES = [1, 2, 3, 4, 5]
DISTANCE_MATRIX = {
    (1, 2): 10, (1, 3): 15, (1, 4): 20, (1, 5): 1,
    (2, 3): 35, (2, 4): 25, (2, 5): 30, (2, 1): 1,
    (3, 4): 30, (3, 5): MAX_INT, (3, 1): 25, (3, 2): 1,
    (4, 5): 15, (4, 1): 30, (4, 2): 25, (4, 3): 1,
    (5, 1): 20, (5, 2): 30, (5, 3): MAX_INT, (5, 4): 1,
}

class Chromosome:
    def __init__(self, genes, fitness_function=sum):
        self.genes = genes
        self.fitness_function = fitness_function
        self.fitness = self.calculate_fitness()

    def to_integer(self, genes):
        return int(''.join([str(bit) for bit in genes]), 2)

    def calculate_fitness(self):
        return self.fitness_function(self.genes)

    def mutate(self):

        for i in range(len(self.genes) - 1):
            if random.random() < MUTATION_RATE:
                # j = random.randint(i + 1, len(self.genes) - 1)
                j, alternate = random.sample(range(len(self.genes)), 2)
                if(i == j):
                    j = alternate
                self.genes[i], self.genes[j] = self.genes[j], self.genes[i]

        self.fitness = self.calculate_fitness()

    def __str__(self):
        return f"Genes: {self.genes}, Fitness: {self.fitness}"
    
    def __eq__(self, other):
        return self.genes == other.genes
    
    def __hash__(self):
        return hash(tuple(self.genes))
    
class Population:
    def __init__(self, size=DEFAULT_POPULATION_SIZE, gene_length=DEFAULT_GENE_LENGTH, fitness_function=sum, initial_city=None):
        self.chromosomes = [self.generate_random_chromosome(gene_length, fitness_function, initial_city) for _ in range(size)]

    @property
    def best_chromosome(self):
        return self.get_best_chromosome()

    # @best_chromosome.setter
    # def best_chromosome(self, value):
    #     self._best_chromosome = value


    def generate_random_chromosome(self, gene_length, fitness_function, initial_city=None):
        if initial_city is not None:
            genes = [initial_city] + random.sample([city for city in CITIES if city != initial_city], gene_length - 1)
        else:
            genes = random.sample(CITIES, gene_length)
        return Chromosome(genes, fitness_function)

    def evolve(self):
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=False)
        parents = self.chromosomes[:len(self.chromosomes)//2]  # Select the top half

        next_generation = []

        for _ in range(len(self.chromosomes)):
            parent1, parent2 = random.choices(parents, k=2)
            
            # child1, child2 = self.crossover(parent1, parent2, 2)
            child1 = self.ordered_crossover(parent1, parent2) 
            child2 = self.ordered_crossover(parent1, parent2) 

            if random.random() < MUTATION_RATE:
                child1.mutate()

            if random.random() < MUTATION_RATE:
                child2.mutate()

            if self.are_genes_valid(child1.genes):
                next_generation.append(child1)

            if self.are_genes_valid(child2.genes):
                next_generation.append(child2)

        self.chromosomes.extend(next_generation)
        self.chromosomes = list(set(self.chromosomes)) # Remove duplicates


    def crossover(self, parent1, parent2, crossover_point=None):
        if crossover_point is None:
            crossover_point = random.randint(1, len(parent1.genes) - 1)

        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]

        return Chromosome(child1_genes, parent1.fitness_function), Chromosome(child2_genes, parent2.fitness_function)
    
    def ordered_crossover(self, parent1, parent2):
        size = len(parent1.genes)
        start, end = sorted(random.sample(range(size), 2))
        child_genes = [None] * size

        child_genes[start:end] = parent1.genes[start:end]

        p2_index = 0
        for i in range(len(child_genes)):
            if child_genes[i] is None:
                while parent2.genes[p2_index] in child_genes:
                    p2_index += 1
                child_genes[i] = parent2.genes[p2_index]

        return Chromosome(child_genes, parent1.fitness_function)
    
    def are_genes_valid(self, genes):
        return sorted(genes) == sorted(CITIES[:len(genes)])

    def get_best_chromosome(self):
        return min(self.chromosomes, key=lambda x: x.fitness)

    def __str__(self):
        return '\n'.join(str(chromosome) for chromosome in self.chromosomes)
    
def example(genes):
    x = int(''.join([str(bit) for bit in genes]), 2)
    return -x**2/10 + 3*x

def tsm_fitness(genes):
    total_distance = 0
    for i in range(len(genes) - 1):
        city1, city2 = genes[i], genes[i + 1]
        total_distance += DISTANCE_MATRIX.get((city1, city2), MAX_INT)
    total_distance += DISTANCE_MATRIX.get((genes[-1], genes[0]), MAX_INT)
    return total_distance
    
if(__name__ == "__main__"):
    population = Population(fitness_function=tsm_fitness, initial_city=5)
    initial = population

    initial_best = population.best_chromosome
    print(initial_best)
    for generation in range(DEFAULT_GENERATION_COUNT):
        population.evolve()
        # print(f"Generation {generation + 1}: Best Chromosome: {population.best_chromosome}, Population Size: {len(population.chromosomes)}")
    
    print("initial population: \n", initial)
    print(f"\nInitial Best Chromosome: {initial_best}")
    print(f"Final Best Chromosome: {population.best_chromosome}")
