import { Injectable } from '@angular/core';
import { ActivatedRouteSnapshot, CanActivate, Router, RouterStateSnapshot, UrlTree } from '@angular/router';
import { AuthService } from './auth.service';

@Injectable({
  providedIn: 'root'
})
export class AuthGuard implements CanActivate {
  constructor(private router: Router, private authService: AuthService){}

  canActivate(): boolean {
    // Vérifiez ici l'état de connexion de l'utilisateur
    const isLoggedIn = false;

    if (this.authService.getisLoggedIn()) {
      return true; // Autorise l'accès à la route
    } else {
      // Redirige l'utilisateur vers la page de connexion
      this.router.navigate(['']);
      return false; // Empêche l'accès à la route
    }
  }
  
}
