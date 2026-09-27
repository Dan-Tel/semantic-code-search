export interface User {
  id: number;
  name: string;
  deleted: boolean;
}

export function sortUsersByName(users: User[]): User[] {
  return [...users].sort((first, second) =>
    first.name.localeCompare(second.name)
  );
}

export const filterDeletedUsers = (
  users: User[]
): User[] => {
  return users.filter((user) => !user.deleted);
};

export function findUserById(
  users: User[],
  userId: number
): User | undefined {
  return users.find((user) => user.id === userId);
}