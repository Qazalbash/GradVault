example_input = "Usama is connected to Saeed, Aaliya, Mohsin.\
Usama traveled to Italy, Japan, Korea.\
Saeed is connected to Sumaira, Zehra, Samar, Marium.\
Saeed traveled to China, Afghanistan.\
Marium is connected to Mohsin, Kashif, Saeed.\
Marium traveled to Japan, USA, Iran.\
Sumaira is connected to Usama, Zehra.\
Sumaira traveled to Japan, Saudi Arabia.\
Aaliya is connected to Mohsin, Bari, Sameera, Kashif.\
Aaliya traveled to India, USA, Malaysia.\
Mohsin is connected to Usama, Bari, Saeed.\
Mohsin traveled to Iran, Indonesia, Afghanistan.\
Bari is connected to Zehra, Usama, Mohsin.\
Bari traveled to Japan, India, China.\
Zehra is connected to Marium, Samar, Saeed.\
Zehra traveled to Russia, Malaysia, Italy.\
Sameera is connected to Bari, Usama, Samar, Kashif.\
Sameera traveled to Afghanistan, Korea, Russia.\
Kashif is connected to Zehra.\
Kashif traveled to Russia, Malaysia.\
Samar is connected to Sumaira, Usama, Aaliya.\
Samar traveled to Saudi Arabia, Indonesia, Iran."


def create_data_structure(string_input):
    lst1 = [
        i.replace(",", "").replace("is", "").replace("connected", "").replace(
            "to", "").replace("traveled", "").split(" ")
        for i in string_input.split(".")
    ]
    lst = [[
        "Saudi Arabia" if k == "Saudi" else k
        for k in j
        if k != "" and k != "Arabia"
    ]
           for j in lst1]
    network = []
    for i in range(0, len(lst), 2):
        try:
            network.append({
                "name": lst[i][0],
                "people": lst[i][1:],
                "countries": lst[i + 1][1:]
            })
        except IndexError:
            network.append({"name": "", "people": "", "countries": ""})
    return network


def get_connections(network, user):
    for pointer in network:
        if pointer["name"] == user:
            return pointer["people"]
    return []


def get_countries_traveled(network, user):
    for pointer in network:
        if pointer["name"] == user:
            return pointer["countries"]
    return []


def add_connection(network, user_A, user_B):
    flag = 0
    for j in network:
        if j["name"] == user_A:
            flag += 1
            if user_B in j["people"]:
                flag += 1
        elif j["name"] == user_B:
            flag += 1
    if flag == 2:
        for i in network:
            if i["name"] == user_A:
                i["people"] = i.get("people", []) + [user_B]
    return network


def add_new_user(network, user, countries):
    network.append({"name": user, "countries": countries})
    return network


def get_secondary_connections(network, user):
    connections = get_connections(network, user)
    connections_of_connections = []
    for connection in connections:
        connections_of_connections += [
            i for i in get_connections(network, connection)
            if i not in connections_of_connections
        ]
    return connections_of_connections


def count_common_connections(network, user_A, user_B):
    count = 0
    connectionsA = get_connections(network, user_A)
    connectionsB = get_connections(network, user_B)
    for person in connectionsA:
        if person in connectionsB:
            count += 1
    return count


def helper2(network, user_A, user_B, discovered, path):
    discovered[user_A] = True
    path.append(user_A)
    if user_A == user_B:
        return path
    for i in get_connections(network, user_A):
        if not (discovered[i]):
            if helper2(network, i, user_B, discovered, path):
                return path
    return None


def find_path_to_patient(network, user_A, user_B):
    discovered = {
        i: False for i in [
            "Usama",
            "Saeed",
            "Marium",
            "Sumaira",
            "Aaliya",
            "Mohsin",
            "Bari",
            "Zehra",
            "Sameera",
            "Kashif",
            "Samar",
            "",
        ]
    }
    return helper2(network, user_A, user_B, discovered, [])


all_paths = []


def helper(network, user_A, user_B, visited, path):
    visited[user_A] = True
    path.append(user_A)
    if user_A == user_B:
        all_paths.append(path.copy())
    else:
        for i in get_connections(network, user_A):
            if visited[i] == False:
                helper(network, i, user_B, visited, path)
    visited[user_A] = False
    path.pop()


def find_all_possible_paths_to_user(network, user_A, user_B):
    path = []
    visited = {
        i: False for i in [
            "Usama",
            "Saeed",
            "Marium",
            "Sumaira",
            "Aaliya",
            "Mohsin",
            "Bari",
            "Zehra",
            "Sameera",
            "Kashif",
            "Samar",
            "",
        ]
    }
    helper(network, user_A, user_B, visited, path)
    return all_paths


net = create_data_structure(example_input)
print(net)
print(get_connections(net, "Aaliya"))
print(get_connections(net, "Marium"))
print(get_countries_traveled(net, "Usama"))
print(add_connection(net, "Usama", "Samar"))
print(add_new_user(net, "Aaliya", []))
print(add_new_user(net, "Nick", ["India", "Italy"]))
print(get_secondary_connections(net, "Marium"))
print(count_common_connections(net, "Marium", "Usama"))
print(find_path_to_patient(net, "Usama", "Zehra"))
print(find_all_possible_paths_to_user(net, "Usama", "Zehra"))
