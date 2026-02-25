# Genetic Algorithm for the Traveling Salesman Problem (TSP) using Holland's Method

import random
from sys import maxsize as MAX_INT

DEFAULT_GENE_LENGTH = 5
DEFAULT_POPULATION_SIZE = 25
DEFAULT_GENERATION_COUNT = 1000

MUTATION_RATE = 0.1

# CITIES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# DISTANCE_MATRIX = {
#     (1, 2): 10, (1, 3): 15, (1, 4): 20, (1, 5): 25, (1, 6): 30, (1, 7): 35, (1, 8): 40, (1, 9): 45, (1, 10): 50,
#     (2, 3): 35, (2, 4): 25, (2, 5): 30, (2, 6): 20, (2, 7): 15, (2, 8): 40, (2, 9): 45, (2, 10): 50, (2, 1): 100,
#     (3, 4): 30, (3, 5): 20, (3, 6): 25, (3, 7): 35, (3, 8): 40, (3, 9): 45, (3, 10): 50, (3, 1): 100, (3, 2): 95,
#     (4, 5): 15, (4, 6): 30, (4, 7): 25, (4, 8): 35, (4, 9): 40, (4, 10): 45, (4, 1): 100, (4, 2): 95, (4, 3): 90,
#     (5, 6): 20, (5, 7): 30, (5, 8): 25, (5, 9): 35, (5, 10): 40, (5, 1): 100, (5, 2): 95, (5, 3): 90, (5, 4): 85,
#     (6, 7): 15, (6, 8): 20, (6, 9): 30, (6, 10): 35, (6, 1): 100, (6, 2): 95, (6, 3): 90, (6, 4): 85, (6, 5): 80,
#     (7, 8): 10, (7, 9): 25, (7, 10): 30, (7, 1): 100, (7, 2): 95, (7, 3): 90, (7, 4): 85, (7, 5): 80, (7, 6): 75,
#     (8, 9): 15, (8, 10): 20, (8, 7): 100, (8, 1): 100, (8, 2): 95, (8, 3): 90, (8, 4): 85, (8, 5): 80, (8, 6): 75,
#     (9, 10): 10, (9, 1): 55, (9, 2): 50, (9, 3): 45, (9, 4): 40, (9, 5): 35, (9, 6): 30, (9, 7): 25, (9, 8): 20,
#     (10, 1): 55, (10, 2): 50, (10, 3): 45, (10, 4): 40, (10, 5): 35, (10, 6): 30, (10, 7): 25, (10, 8): 20, (10, 9): 15
# }

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
        index = random.randint(0, len(self.genes) - 1)
        city, alternate = random.sample(self.genes, 2)
        self.genes[index] = city if self.genes[index] != city else alternate
        self.fitness = self.calculate_fitness()

    def __str__(self):
        return f"Genes: {self.genes}, Fitness: {self.fitness}"
    
    def __eq__(self, other):
        return self.genes == other.genes
    
    def __hash__(self):
        return hash(tuple(self.genes))
    
class Population:
    def __init__(self, size=DEFAULT_POPULATION_SIZE, gene_length=DEFAULT_GENE_LENGTH, fitness_function=sum, initial_city=None):
        self.chromosomes = [self.generate_random_chromosome(gene_length, fitness_function, initial_city) for _ in range(size-2)]
        self.chromosomes.append(Chromosome([5, 4, 1, 2, 3], fitness_function))
        self.chromosomes.append(Chromosome([4, 5, 3, 2, 1], fitness_function))
        self.best_chromosome = self.get_best_chromosome()

    def generate_random_chromosome(self, gene_length, fitness_function, initial_city=None):
        if initial_city is not None:
            genes = [initial_city] + random.sample([city for city in CITIES if city != initial_city], gene_length - 1)
        else:
            genes = random.sample(CITIES, gene_length)
        return Chromosome(genes, fitness_function)

    def evolve(self):
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=False)
        parents = self.chromosomes[:len(self.chromosomes)//2]  # Select the top half
        print("Size of current population:", len(self.chromosomes))
        print("Size of parents:", len(parents))

        next_generation = []

        for _ in range(len(self.chromosomes)):
            parent1, parent2 = random.choices(parents, k=2)
            
            child1, child2 = self.crossover(parent1, parent2, 2) 

            if random.random() < MUTATION_RATE:
                child1.mutate()

            if random.random() < MUTATION_RATE:
                child2.mutate()

            if self.are_genes_valid(child1.genes):
                next_generation.append(child1)
                # print("Valid child: ", child1)

            if self.are_genes_valid(child2.genes):
                next_generation.append(child2)
                # print("Valid child: ", child2)

            # print(next_generation)

        self.chromosomes.extend(next_generation)
        self.chromosomes = list(set(self.chromosomes)) # Remove duplicates
        # print([x.genes for x in self.chromosomes])
        best = self.get_best_chromosome()
        if self.best_chromosome is None or best.fitness > self.best_chromosome.fitness:
            self.best_chromosome = best


    def crossover(self, parent1, parent2, crossover_point=None):
        # print("p1", parent1.genes, "p2", parent2.genes)
        if crossover_point is None:
            crossover_point = random.randint(1, len(parent1.genes) - 1)

        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]

        return Chromosome(child1_genes, parent1.fitness_function), Chromosome(child2_genes, parent2.fitness_function)
    
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

    print("Initial Population:")
    print(population)
    print(f"Best Chromosome: {population.best_chromosome}")
    print()

    initial_best = population.best_chromosome
    for generation in range(DEFAULT_GENERATION_COUNT):
        population.evolve()
        print(f"Generation {generation + 1}: Best Chromosome: {population.best_chromosome}, Population Size: {len(population.chromosomes)}")
    
    print("initial population: \n", initial)
    print(f"\nInitial Best Chromosome: {initial_best}")
    print(f"Final Best Chromosome: {population.best_chromosome}")
