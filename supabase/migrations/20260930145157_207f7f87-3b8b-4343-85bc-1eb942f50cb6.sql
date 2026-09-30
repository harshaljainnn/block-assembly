create type public.app_role as enum ('manager', 'operator');

create table public.user_roles (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  role public.app_role not null,
  unique (user_id, role)
);
grant select on public.user_roles to authenticated;
grant all on public.user_roles to service_role;
alter table public.user_roles enable row level security;

create or replace function public.has_role(_user_id uuid, _role public.app_role)
returns boolean language sql stable security definer set search_path = public
as $$ select exists (select 1 from public.user_roles where user_id = _user_id and role = _role) $$;

create policy "Users read own roles" on public.user_roles for select to authenticated
using (user_id = auth.uid() or public.has_role(auth.uid(), 'manager'));

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  name text not null default '',
  email text not null default '',
  goal_per_day integer not null default 15,
  created_at timestamptz not null default now()
);
grant select, update on public.profiles to authenticated;
grant all on public.profiles to service_role;
alter table public.profiles enable row level security;

create policy "Read own or manager reads all" on public.profiles for select to authenticated
using (id = auth.uid() or public.has_role(auth.uid(), 'manager'));
create policy "Managers update profiles" on public.profiles for update to authenticated
using (public.has_role(auth.uid(), 'manager')) with check (public.has_role(auth.uid(), 'manager'));

create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public
as $$
begin
  insert into public.profiles (id, name, email)
  values (new.id, coalesce(new.raw_user_meta_data->>'name', split_part(new.email, '@', 1)), new.email);
  if not exists (select 1 from public.user_roles where role = 'manager') then
    insert into public.user_roles (user_id, role) values (new.id, 'manager');
  else
    insert into public.user_roles (user_id, role) values (new.id, 'operator');
  end if;
  return new;
end $$;

create trigger on_auth_user_created after insert on auth.users
for each row execute function public.handle_new_user();